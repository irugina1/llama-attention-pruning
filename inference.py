from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
from torch import nn
import numpy as np
from datasets import load_dataset
import modal
from modal import Stub, Image, gpu
import gc
from typing import Optional, Tuple
from deepspeed.ops.sparse_attention import SparseSelfAttention, DenseSparsityConfig

# constants
DEVICE = "cuda"
MODEL_PATH = "Llama-2-7b-chat-hf"
MAX_LENGTH = 4096  # context length for llama2
STRIDE = 4096
BATCH_SIZE = 4
USE_SPARSE_ATTENTION = False

# modal setup
stub = Stub()
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "cmake", "libncurses5-dev", "libncursesw5-dev", "zlib1g-dev")
    .pip_install(
        "torch==1.13.0",
        "transformers==4.28.0",
        "numpy==1.26.3",
        "datasets==2.17.0",
        "deepspeed==0.9.0",
    )
    .run_commands(
        "git clone https://github.com/openai/triton.git && "
        "cd triton && "
        "git checkout 44442db96ef6dc55d27dc047ce240d0b1397e5ef && "
        "cd python && "
        # "pip install cmake && cmake --version && "
        "pip install -e ."
    )
)
GPU_CONFIG = gpu.A100(memory=80, count=1)
volume = modal.Volume.from_name("llama-2-7b-chat-hf")


def running_average(old_average, sum_new_values, N, M):
    return old_average * N / (N + M) + (sum_new_values) / (N + M)


def update_nll_running_average(model, input_ids, target_ids, average, N):
    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss
    bsz = input_ids.shape[0]
    average = running_average(average, neg_log_likelihood.item() * bsz, N, bsz)
    N += bsz
    return average, N


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to
        # obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`.
        # Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to
            # obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :], persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaSparseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, sparsity_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings
        )

        self.sparse_self_attention = SparseSelfAttention(
            sparsity_config, max_seq_length=4096, attn_mask_mode='add'
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_output = self.sparse_self_attention(
            query_states, key_states, value_states, attn_mask=attention_mask
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        attn_weights = None

        return attn_output, attn_weights, past_key_value


def replace_self_attention_layers(config, layers, sparsity_config):
    for layer in layers:
        deepspeed_sparse_self_attn = LlamaSparseAttention(config, sparsity_config).to(
            DEVICE
        )
        deepspeed_sparse_self_attn.q_proj = layer.self_attn.q_proj
        deepspeed_sparse_self_attn.k_proj = layer.self_attn.k_proj
        deepspeed_sparse_self_attn.v_proj = layer.self_attn.v_proj
        deepspeed_sparse_self_attn.o_proj = layer.self_attn.o_proj
        deepspeed_sparse_self_attn.rotary_emb = layer.self_attn.rotary_emb

        layer.self_attn = deepspeed_sparse_self_attn

    return layers


@stub.function(
    volumes={"/my_vol": volume}, image=image, gpu=GPU_CONFIG, timeout=60 * 60 * 24
)
def main():
    model_path = f"/my_vol/{MODEL_PATH}/"
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, cache_dir="llm_weights")
        .half()
        .to(DEVICE)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf", cache_dir="llm_weights", use_fast=True
    )

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    config = model.config
    sparsity_config = DenseSparsityConfig(num_heads=32)
    if USE_SPARSE_ATTENTION:
        print("Using deepspeed sparse attention")
        replace_self_attention_layers(config, model.model.layers, sparsity_config)

    prev_end_loc = 0
    average = 0
    N = 0
    batch_counter = 0
    num_batches = math.ceil(seq_len/(STRIDE * BATCH_SIZE))
    for outer_begin_loc in range(0, seq_len, STRIDE * BATCH_SIZE):
        batch_counter += 1
        print(f"starting work on batch {batch_counter} out of {num_batches}")
        batch_input_ids = []
        batch_target_ids = []
        for begin_loc in range(
            outer_begin_loc, outer_begin_loc + STRIDE * BATCH_SIZE, STRIDE
        ):
            end_loc = min(begin_loc + MAX_LENGTH, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            prev_end_loc = end_loc
            batch_input_ids.append(input_ids)
            batch_target_ids.append(target_ids)
            if end_loc == seq_len:
                break

        if batch_input_ids[-1].shape[-1] != MAX_LENGTH:
            num_sequences = len(batch_input_ids)
            for sequence_idx in range(0, num_sequences):
                input_ids = batch_input_ids[sequence_idx]
                target_ids = batch_target_ids[sequence_idx]
                with torch.inference_mode():
                    average, N = update_nll_running_average(
                        model, input_ids, target_ids, average, N)

        else:
            input_ids = torch.squeeze(torch.stack(batch_input_ids), 1)
            target_ids = torch.squeeze(torch.stack(batch_target_ids), 1)

            with torch.inference_mode():
                average, N = update_nll_running_average(
                    model, input_ids, target_ids, average, N)
        print(average)
    ppl = np.exp(average)
    print(f"estimated perplexity = {ppl}")


if __name__ == "__main__":
    main()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
import modal
from modal import Stub, Image, gpu
import gc

# constants
# mps was actually faster than cuda for this setup
LOCAL = True
DEVICE = "mps" if LOCAL else "cuda"
MODEL_PATH = "Llama-2-7b-chat-hf"
MAX_LENGTH = 4096  # context length for llama2
STRIDE = 4096

# modal setup
stub = Stub()
image = Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
).pip_install(
    "torch==2.1.2", "transformers==4.28.0", "numpy==1.26.3", "datasets==2.17.0"
)
GPU_CONFIG = gpu.A100(memory=80, count=1)
volume = modal.Volume.from_name("llama-2-7b-chat-hf")


def running_average(average, new_values, N):
    new_N = N + 1
    average *= (N / new_N)
    average += (new_values / new_N)
    return average, new_N


def attention_patterns_per_chunk(model, encodings, average, N):
    seq_len = encodings.input_ids.size(1)
    print(f"seq len = {seq_len}")

    for begin_loc in range(0, seq_len, STRIDE):
        print(f"  {begin_loc / seq_len} done")
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        if end_loc == seq_len:
            break

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        with torch.inference_mode():
            outputs = model(input_ids)
            attention_outputs = [
                x.cpu().detach().numpy() for x in outputs["attentions"]
            ]
            all_attn = np.stack(attention_outputs)
            all_attn = np.squeeze(all_attn)
            print(
                f"all_attn shape, memory footprint, and dtype = {all_attn.shape}, {all_attn.nbytes >> 30} GB, {all_attn.dtype}"  # noqa
            )

            average, N = running_average(average, all_attn, N)
            del outputs
            del attention_outputs
            del all_attn
            torch.mps.empty_cache()
            gc.collect()

    return average, N


def checkpoint(average, counts, checkpoint_descriptor):
    save_dir = "." if LOCAL else "/my_vol"
    with open(f"{save_dir}/average_attention_{checkpoint_descriptor}.npy", "wb") as f:
        np.save(f, average)
    with open(f"{save_dir}/counts_{checkpoint_descriptor}.npy", "wb") as f:
        np.save(f, counts)
    if not LOCAL:
        volume.commit()


@stub.function(
    volumes={"/my_vol": volume}, image=image, gpu=GPU_CONFIG, timeout=60 * 60 * 24
)
def main():
    model_path = f"./{MODEL_PATH}/" if LOCAL else f"/my_vol/{MODEL_PATH}/"

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir="llm_weights", output_attentions=True
        )
        .half()
        .to(DEVICE)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf", cache_dir="llm_weights", use_fast=True
    )

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    print("model size: {:.3f}GB".format(param_size >> 30))

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    NUM_CHUNKS = 20
    boundaries = np.arange(
        0, len(train["text"]), int(len(train["text"]) / NUM_CHUNKS), dtype=int
    )
    boundaries[-1] = len(train["text"])

    average = np.zeros((32, 32, 4096, 4096), dtype=np.float16)
    print(
        f"average shape, memory footprint, and dtype = {average.shape}, {average.nbytes >> 30} GB, {average.dtype}"  # noqa
    )
    N = 0
    chunk_idx = 0
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        print(f"starting chunk {chunk_idx + 1} out of {NUM_CHUNKS}")
        encodings = tokenizer(
            " ".join(train["text"][start_idx:end_idx]), return_tensors="pt"
        )
        average, N = attention_patterns_per_chunk(model, encodings, average, N)
        checkpoint(average, N, chunk_idx + 1)
        chunk_idx += 1
    checkpoint(average, N, "final")


if __name__ == "__main__":
    main.local()

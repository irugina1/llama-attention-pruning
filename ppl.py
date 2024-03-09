from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
import math

# constants
DEVICE = "mps"
MODEL_ID = "Llama-2-7b-chat-hf"
MAX_LENGTH = 4096  # context length for llama2
STRIDE = 4096
BATCH_SIZE = 2  # need one to collect attention patterns


def running_average(old_average, sum_new_values, N, M):
    return old_average * N / (N + M) + (sum_new_values) / (N + M)


def update_nll_running_average(model, input_ids, target_ids, average, N):
    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss
    bsz = input_ids.shape[0]
    average = running_average(average, neg_log_likelihood.item() * bsz, N, bsz)
    N += bsz
    return average, N


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir="llm_weights").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf", cache_dir="llm_weights", use_fast=True
    )

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    # seq len
    seq_len = encodings.input_ids.size(1)
    print(f"seq len = {seq_len}")
    # init running variables
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

    ppl = np.exp(average)
    print(f"estimated perplexity = {ppl}")


if __name__ == "__main__":
    main()

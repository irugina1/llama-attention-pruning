import numpy as np
import torch
import argparse


def get_sparsity_config(attention, percentile, block_size):
    threshold = np.percentile(attention, percentile, axis=(2, 3), keepdims=True)
    comparison = attention > threshold
    sparsity = (
        comparison.reshape(
            comparison.shape[0],
            comparison.shape[1],
            comparison.shape[2] // block_size,
            block_size,
            comparison.shape[3] // block_size,
            block_size,
        )
        .all(axis=(-1, -3))
    )
    return sparsity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", default=16)
    parser.add_argument("--attention_path", required=True)
    parser.add_argument("--percentile", type=int, required=True)
    args = parser.parse_args()

    attention = np.load(args.attention_path)
    sparsity = get_sparsity_config(attention, args.percentile, args.block_size)
    tensor = torch.tensor(sparsity)
    torch.save(
        tensor,
        f'layout_{args.attention_path.split(".")[0]}_percentile_{args.percentile}.pt',
    )

import numpy as np


def get_sparsity_config(attention, percentile, block_size):
    threshold = np.percentile(attention, percentile + 1, axis=(2, 3), keepdims=True)
    comparison = attention < threshold
    sparsity = comparison.reshape(
        comparison.shape[0],
        comparison.shape[1],
        comparison.shape[2] // block_size,
        block_size,
        comparison.shape[3] // block_size,
        block_size,
    ).all(axis=(-1, -3))
    return sparsity

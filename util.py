import numpy as np
from typing import List
from warnings import warn


def split_data_set(x: List, y: List, ratio: List[float] = None):
    if ratio is None:
        ratio = [0.5, 0.3]
    if len(x) != len(y):
        raise AssertionError('dataset lists are not the same length')
    if np.sum(ratio) > 1:
        raise AssertionError('The sum of the ratios must not exceed 1')
    if any(r*len(x) < 1 for r in ratio):
        warn('Too little ratio')

    indices = np.multiply(np.cumsum(ratio), len(x)).astype(np.int32)
    split = [(x[:indices[0]], y[:indices[0]])]
    for i, j in zip(indices[:-1], indices[1:]):
        split.append((x[i:j], y[i:j]))
    split.append((x[indices[-1]:], y[indices[-1]:]))

    return split


# def print_data(x: List):

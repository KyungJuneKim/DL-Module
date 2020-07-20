import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import History
from typing import List
from warnings import warn


def split_data_set(x: List, y: List, ratio: List[float]):
    if not ratio:
        raise AttributeError('Nothing in ratio')
    if len(x) != len(y):
        raise AssertionError('lists, x and y, are not the same length')
    if np.sum(ratio) > 1:
        raise AssertionError('The sum of the ratios must not exceed 1')
    if any(r*len(x) < 1 for r in ratio):
        warn('Too little ratio')

    error = int(np.around(np.log10(len(x)))) - 15
    indices = np.around(np.multiply(np.cumsum(ratio), len(x)), -error).astype(np.int32)
    split = [(x[:indices[0]], y[:indices[0]])]
    for i, j in zip(indices[:-1], indices[1:]):
        split.append((x[i:j], y[i:j]))
    split.append((x[indices[-1]:], y[indices[-1]:]))

    return split


def plot_model(h: History, validation: bool = False, keys: List[str] = None):
    if not keys:
        keys = ['loss']
    fig, axes = plt.subplots(nrows=1, ncols=len(keys), sharex='all', figsize=(15, 6))

    for idx, key in enumerate(keys):
        axes[idx].set_title('Model ' + key)
        axes[idx].plot(h.history[key], label=key)
        if validation:
            axes[idx].plot(h.history['val_' + key], label='val_' + key)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(key.capitalize())
        axes[idx].legend()

    fig.tight_layout()
    plt.show()


def print_data(x: List):
    plt.plot(x)
    plt.tight_layout()
    plt.show()

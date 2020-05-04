"""
PyTorch commonly used functions.
"""
import numpy as np
import torch as t

def default_device() -> t.device:
    """
    PyTorch default device is GPU when available, CPU otherwise.

    :return: Default device.
    """
    return t.device('cuda' if t.cuda.is_available() else 'cpu')

def to_tensor(array: np.ndarray) -> t.Tensor:
    """
    Convert numpy array to tensor on default device.

    :param array: Numpy array to convert.
    :return: PyTorch tensor on default device.
    """
    return t.tensor(array, dtype=t.float32).to(default_device())

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result
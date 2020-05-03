import inspect
import os

import numpy as np
import torch as t

from common.settings import STORAGE

def to_tensor(array: np.ndarray, use_cuda: bool = True):
    tensor = t.tensor(array, dtype=t.float32)
    if use_cuda and t.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def to_device(module: t.nn.Module, use_cuda: bool = True):
    return module.cuda() if use_cuda and t.cuda.is_available() else module


def div_no_nan(a, b):
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def get_module_path():
    module_path = os.path.dirname(inspect.stack()[1].filename)
    return module_path

def experiment_path(module_path: str, name: str):
    return os.path.join(STORAGE, module_path, name)

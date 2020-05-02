"""
Loss functions for PyTorch.
"""

import torch as t

from common.torch.ops import divide_no_nan

def mape_loss(forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    weights = divide_no_nan(mask, target)
    return t.mean(t.abs((forecast - target) * weights))

def smape_1_loss(forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    return 200 * t.mean(divide_no_nan(t.abs(forecast - target), forecast.data + target.data) * mask)


def smape_2_loss(forecast, target, mask) -> t.float:
    """
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                      t.abs(forecast.data) + t.abs(target.data)) * mask)


def mase_loss(insample: t.Tensor, freq: int,
              forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param insample: Insample values. Shape: batch, time_i
    :param freq: Frequency value
    :param forecast: Forecast values. Shape: batch, time_o
    :param target: Target values. Shape: batch, time_o
    :param mask: 0/1 mask. Shape: batch, time_o
    :return: Loss value
    """
    masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
    masked_masep_inv = divide_no_nan(mask, masep[:, None])
    return t.mean(t.abs(target - forecast) * masked_masep_inv)
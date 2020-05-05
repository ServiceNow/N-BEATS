"""
Shortcut functions to create N-BEATS models.
"""
import gin
import numpy as np
import torch as t

from models.nbeats import GenericBasis, NBeats, NBeatsBlock, SeasonalityBasis, TrendBasis


@gin.configurable()
def interpretable(input_size: int,
                  output_size: int,
                  trend_blocks: int,
                  trend_layers: int,
                  trend_layer_size: int,
                  degree_of_polynomial: int,
                  seasonality_blocks: int,
                  seasonality_layers: int,
                  seasonality_layer_size: int,
                  num_of_harmonics: int):
    """
    Create N-BEATS interpretable model.
    """
    trend_block = NBeatsBlock(input_size=input_size,
                              theta_size=2 * (degree_of_polynomial + 1),
                              basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        backcast_size=input_size,
                                                        forecast_size=output_size),
                              layers=trend_layers,
                              layer_size=trend_layer_size)
    seasonality_block = NBeatsBlock(input_size=input_size,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    backcast_size=input_size,
                                                                    forecast_size=output_size),
                                    layers=seasonality_layers,
                                    layer_size=seasonality_layer_size)

    return NBeats(t.nn.ModuleList(
        [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]))


@gin.configurable()
def generic(input_size: int, output_size: int,
            stacks: int, layers: int, layer_size: int):
    """
    Create N-BEATS generic model.
    """
    return NBeats(t.nn.ModuleList([NBeatsBlock(input_size=input_size,
                                               theta_size=input_size + output_size,
                                               basis_function=GenericBasis(backcast_size=input_size,
                                                                           forecast_size=output_size),
                                               layers=layers,
                                               layer_size=layer_size)
                                   for _ in range(stacks)]))

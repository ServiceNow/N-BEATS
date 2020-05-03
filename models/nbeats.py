from typing import Tuple

import numpy as np
import torch as t


class NBeatsFC(t.nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_layers: int,
                 output_size: int):
        super().__init__()
        self.fc_layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=output_size)] +
                                         [t.nn.Linear(in_features=output_size, out_features=output_size)
                                          for _ in range(fc_layers - 1)])

    def forward(self, x: t.Tensor) -> t.Tensor:
        output = x
        for layer in self.fc_layers:
            output = t.relu(layer(output))
        return output


class NBeatsGenericBlock(t.nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_layers: int,
                 fc_layers_size: int,
                 output_size: int):
        super().__init__()
        self.fc = NBeatsFC(input_size=input_size,
                           fc_layers=fc_layers,
                           output_size=fc_layers_size)
        self.basis = t.nn.Linear(in_features=fc_layers_size, out_features=input_size + output_size)

        self.backcast_dump = None
        self.forecast_dump = None

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        input_size = x.shape[1]
        output = self.basis(self.fc(x))
        backcast = output[:, :input_size]
        forecast = output[:, input_size:]
        self.backcast_dump = backcast
        self.forecast_dump = forecast
        return backcast, forecast


class NBeatsTrendBlock(t.nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_layers: int,
                 fc_layers_size: int,
                 degree_of_polynomial: int,
                 output_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.fc = NBeatsFC(input_size=input_size,
                           fc_layers=fc_layers,
                           output_size=fc_layers_size)
        self.basis = t.nn.Linear(in_features=fc_layers_size, out_features=2 * self.polynomial_size)
        self.output_size = output_size
        self.backcast_time = np.concatenate([np.power(np.arange(input_size, dtype=np.float) / input_size, i)[None, :]
                                             for i in range(self.polynomial_size)])
        self.forecast_time = np.concatenate([np.power(np.arange(output_size, dtype=np.float) / output_size, i)[None, :]
                                             for i in range(self.polynomial_size)])
        self.backcast_dump = None
        self.forecast_dump = None

    def forward(self, x: t.Tensor):
        thetas = self.basis(self.fc(x))
        backcast = t.einsum('bp,pt->bt', thetas[:, self.polynomial_size:], x.new(self.backcast_time))
        forecast = t.einsum('bp,pt->bt', thetas[:, :self.polynomial_size], x.new(self.forecast_time))
        self.backcast_dump = backcast
        self.forecast_dump = forecast
        return backcast, forecast


class NBeatsSeasonalityBlock(t.nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_layers: int,
                 fc_layers_size: int,
                 num_of_harmonics: int,
                 output_size: int):
        super().__init__()
        self.basis_parameters = int(np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1))

        self.fc = NBeatsFC(input_size=input_size,
                           fc_layers=fc_layers,
                           output_size=fc_layers_size)
        self.basis = t.nn.Linear(in_features=fc_layers_size, out_features=4 * self.basis_parameters)

        frequency = np.append(np.zeros(1, dtype=np.float32),
                              np.arange(num_of_harmonics, num_of_harmonics / 2 * output_size,
                                        dtype=np.float32) / num_of_harmonics)[None, :]
        backcast_grid = -2 * np.pi * (np.arange(input_size, dtype=np.float32)[:, None] / output_size) * frequency
        forecast_grid = 2 * np.pi * (np.arange(output_size, dtype=np.float32)[:, None] / output_size) * frequency
        self.backcast_cos_template = np.transpose(np.cos(backcast_grid))
        self.backcast_sin_template = np.transpose(np.sin(backcast_grid))
        self.forecast_cos_template = np.transpose(np.cos(forecast_grid))
        self.forecast_sin_template = np.transpose(np.sin(forecast_grid))

        self.backcast_dump = None
        self.forecast_dump = None

    def forward(self, x: t.Tensor):
        harmonics_weights = self.basis(self.fc(x))

        backcast_harmonics_cos = t.einsum('bp,pt->bt',
                                          harmonics_weights[:, 2 * self.basis_parameters:3 * self.basis_parameters],
                                          x.new(self.backcast_cos_template))
        backcast_harmonics_sin = t.einsum('bp,pt->bt',
                                          harmonics_weights[:, 3 * self.basis_parameters:],
                                          x.new(self.backcast_sin_template))
        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          harmonics_weights[:, :self.basis_parameters],
                                          x.new(self.forecast_cos_template))
        forecast_harmonics_sin = t.einsum('bp,pt->bt',
                                          harmonics_weights[:, self.basis_parameters:2 * self.basis_parameters],
                                          x.new(self.forecast_sin_template))
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        self.backcast_dump = backcast
        self.forecast_dump = forecast

        return backcast, forecast


class NBeats(t.nn.Module):
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


def nbeats_generic(input_size: int, output_size: int,
                   blocks: int = 30, fc_layers: int = 4, fc_layers_size: int = 512):
    return NBeats(t.nn.ModuleList([NBeatsGenericBlock(input_size=input_size,
                                                      fc_layers=fc_layers,
                                                      fc_layers_size=fc_layers_size,
                                                      output_size=output_size)
                                   for _ in range(blocks)]))


def nbeats_interpretable(input_size: int, output_size: int,
                         trend_blocks: int = 3,
                         trend_fc_layers: int = 4,
                         trend_fc_layers_size: int = 256,
                         degree_of_polynomial: int = 3,
                         seasonality_blocks: int = 3,
                         seasonality_fc_layers: int = 4,
                         seasonality_fc_layers_size: int = 2048,
                         num_of_harmonics: int = 1):
    trend_block = NBeatsTrendBlock(input_size=input_size,
                                   fc_layers=trend_fc_layers,
                                   fc_layers_size=trend_fc_layers_size,
                                   degree_of_polynomial=degree_of_polynomial,
                                   output_size=output_size)
    seasonality_block = NBeatsSeasonalityBlock(input_size=input_size,
                                               fc_layers=seasonality_fc_layers,
                                               fc_layers_size=seasonality_fc_layers_size,
                                               num_of_harmonics=num_of_harmonics,
                                               output_size=output_size)
    return NBeats(t.nn.ModuleList(
        [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]))
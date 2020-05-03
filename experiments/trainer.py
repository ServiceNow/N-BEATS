import os
from typing import Iterator

import numpy as np
import torch as t
from torch import optim

from common.torch.snapshots import SnapshotManager
from common.torch.losses import smape_2_loss, mape_loss, mase_loss
from experiments.utils import to_tensor, to_device
from experiments.parameters import Parameters
from models.nbeats import nbeats_generic, nbeats_interpretable


def train_nbeats(experiment_path: str,
                 input_size: int,
                 output_size: int,
                 seasonal_pattern: str,
                 experiment_parameters: Parameters,
                 training_set: Iterator,
                 timeseries_frequency: int):
    snapshot_dir = os.path.join(experiment_path, 'snapshots', seasonal_pattern)

    snapshot_manager = SnapshotManager(snapshot_dir=snapshot_dir,
                                       logging_frequency=experiment_parameters.logging_frequency_for(seasonal_pattern),
                                       snapshot_frequency=experiment_parameters.snapshot_frequency_for(
                                           seasonal_pattern))

    model = nbeats_generic(input_size=input_size,
                           output_size=output_size,
                           blocks=experiment_parameters.generic_blocks,
                           fc_layers=experiment_parameters.fc_layers,
                           fc_layers_size=experiment_parameters.generic_fc_layers_size,
                           ) if experiment_parameters.model_type == 'generic' \
        else nbeats_interpretable(input_size=input_size,
                                  output_size=output_size,
                                  trend_blocks=experiment_parameters.trend_blocks,
                                  trend_fc_layers=experiment_parameters.fc_layers,
                                  trend_fc_layers_size=experiment_parameters.trend_fc_layers_size,
                                  degree_of_polynomial=experiment_parameters.degree_of_polynomial,
                                  seasonality_blocks=experiment_parameters.seasonality_blocks,
                                  seasonality_fc_layers=experiment_parameters.fc_layers,
                                  seasonality_fc_layers_size=experiment_parameters.seasonality_fc_layers_size,
                                  num_of_harmonics=experiment_parameters.num_of_harmonics)

    model = to_device(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=experiment_parameters.learning_rate,
                           weight_decay=experiment_parameters.weight_decay)

    training_loss_fn = __loss_fn(experiment_parameters.loss_name)

    iterations = experiment_parameters.iterations_for(seasonal_pattern)

    lr_decay_step = iterations // 3
    if lr_decay_step == 0:
        lr_decay_step = 1

    iteration = snapshot_manager.restore(model, optimizer)

    #
    # Training Loop
    #
    snapshot_manager.enable_time_tracking()
    training_set = iter(training_set)
    for i in range(iteration + 1, iterations + 1):
        model.train()
        x, x_mask, y, y_mask = map(to_tensor, next(training_set))
        optimizer.zero_grad()
        forecast = model(x, x_mask)
        training_loss = training_loss_fn(x, timeseries_frequency, forecast, y, y_mask)

        if np.isnan(float(training_loss)):
            break

        training_loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = experiment_parameters.learning_rate * 0.5 ** (i // lr_decay_step)

        snapshot_manager.register(iteration=i,
                                  training_loss=float(training_loss),
                                  validation_loss=np.nan, model=model,
                                  optimizer=optimizer)
    return model


def __loss_fn(loss_name: str):
    def loss(x, freq, forecast, target, target_mask):
        if loss_name == 'MAPE':
            return mape_loss(forecast, target, target_mask)
        elif loss_name == 'MASE':
            return mase_loss(x, freq, forecast, target, target_mask)
        elif loss_name == 'SMAPE':
            return smape_2_loss(forecast, target, target_mask)
        else:
            raise Exception(f'Unknown loss function: {loss_name}')

    return loss
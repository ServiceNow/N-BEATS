import logging
import os

import numpy as np
import pandas as pd
import torch as t
from fire import Fire

from datasets.m3 import M3Dataset, M3Meta
from experiments.experiment import create_experiment
from experiments.m3.parameters import parameters
from experiments.parameters import Parameters
from experiments.samplers import UnivariateTimeseriesSampler
from experiments.trainer import train_nbeats
from experiments.utils import get_module_path, experiment_path
from experiments.utils import to_tensor
from summary.utils import group_values

module_path = get_module_path()


def init(name: str):
    create_experiment(experiment_path=experiment_path(module_path, name),
                      parameters=parameters[name],
                      command=lambda path, params: f'python {module_path}/main.py run --path={path}')


def run(path: str):
    experiment_parameters = Parameters.load(path)

    dataset = M3Dataset.load(training=True)

    forecasts = []
    for seasonal_pattern in M3Meta.seasonal_patterns:
        history_size_in_horizons = experiment_parameters.history_size_for(seasonal_pattern)
        horizon = M3Meta.horizons_map[seasonal_pattern]
        input_size = experiment_parameters.input_size * horizon

        # Training Set
        training_values = group_values(dataset.values, dataset.groups, seasonal_pattern)

        if experiment_parameters.validation_mode:
            training_values = np.array([v[:-horizon] for v in training_values])

        training_set = UnivariateTimeseriesSampler(timeseries=training_values,
                                                   insample_size=input_size,
                                                   outsample_size=horizon,
                                                   window_sampling_limit=int(history_size_in_horizons * horizon),
                                                   batch_size=experiment_parameters.training_batch_size)

        # Create/restore trained model
        model = train_nbeats(experiment_path=path,
                             input_size=input_size,
                             output_size=horizon,
                             seasonal_pattern=seasonal_pattern,
                             experiment_parameters=experiment_parameters,
                             training_set=iter(training_set),
                             timeseries_frequency=M3Meta.frequency_map[seasonal_pattern])

        #
        # Predict
        #
        x, x_mask = map(to_tensor, training_set.sequential_latest_insamples())
        model.eval()
        with t.no_grad():
            forecasts.extend(model(x, x_mask).cpu().detach().numpy())

    forecasts_df = pd.DataFrame(forecasts, columns=[f'V{i + 1}' for i in range(np.max(M3Meta.horizons))])
    forecasts_df.index = dataset.ids
    forecasts_df.index.name = 'id'
    forecasts_df.to_csv(os.path.join(path, 'forecast.csv'))


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()
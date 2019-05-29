# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Traffic Experiment
"""
import logging
import os
from typing import Dict

import gin
import numpy as np
import pandas as pd
import torch as t
from fire import Fire

from common.experiment import Experiment
from common.sampler import TimeseriesSampler
from common.torch.ops import to_tensor
from common.torch.snapshots import SnapshotManager
from datasets.traffic import TrafficDataset, TrafficMeta
from experiments.model import generic, interpretable
from experiments.trainer import trainer

splits = {
    # https://arxiv.org/pdf/1704.04110.pdf
    'deepar': '2008-06-14 23',
    # Figure B.5 http://proceedings.mlr.press/v97/wang19k/wang19k-supp.pdf
    'deepfactors': '2008-01-13 23',
    # the last 7 days according to our holidays assumption
    'last': '2009-03-24 00'
}
test_windows = 7


class TrafficExperiment(Experiment):
    @gin.configurable()
    def instance(self,
                 repeat: int,
                 lookback: int,
                 loss: str,
                 history_size: int,
                 iterations: Dict[str, int],
                 model_type: str):
        dataset = TrafficDataset.load()

        for split_name, split_date in splits.items():
            horizon = TrafficMeta.horizon
            input_size = lookback * horizon

            # Training Set
            training_set, test_set = dataset.split_by_date(split_date)
            training_values = training_set.values
            test_values = test_set.values

            training_set = TimeseriesSampler(timeseries=training_values,
                                             insample_size=input_size,
                                             outsample_size=horizon,
                                             window_sampling_limit=history_size * horizon)

            if model_type == 'interpretable':
                model = interpretable(input_size=input_size, output_size=horizon)
            elif model_type == 'generic':
                model = generic(input_size=input_size, output_size=horizon)
            else:
                raise Exception(f'Unknown model type {model_type}')

            # Train model
            snapshot_manager = SnapshotManager(snapshot_dir=os.path.join(self.root, 'snapshots', split_name),
                                               total_iterations=iterations[split_name])
            model = trainer(snapshot_manager=snapshot_manager,
                            model=model,
                            training_set=iter(training_set),
                            timeseries_frequency=TrafficMeta.frequency,
                            loss_name=loss,
                            iterations=iterations[split_name])

            # Build forecasts
            forecasts = []
            model.eval()
            with t.no_grad():
                for i in range(test_windows):
                    window_input_set = np.concatenate([training_values, test_values[:, :i * TrafficMeta.horizon]],
                                                      axis=1)
                    input_set = TimeseriesSampler(timeseries=window_input_set,
                                                  insample_size=input_size,
                                                  outsample_size=horizon,
                                                  window_sampling_limit=int(
                                                      history_size * horizon))
                    x, x_mask = map(to_tensor, input_set.last_insample_window())
                    window_forecast = model(x, x_mask).cpu().detach().numpy()
                    forecasts = window_forecast if len(forecasts) == 0 else np.concatenate([forecasts, window_forecast],
                                                                                           axis=1)

            forecasts_df = pd.DataFrame(forecasts, columns=[f'V{i + 1}'
                                                            for i in range(test_windows * TrafficMeta.horizon)])
            forecasts_df.index.name = 'id'
            forecasts_df.to_csv(os.path.join(self.root, f'forecast_{split_name}.csv'))


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(TrafficExperiment)

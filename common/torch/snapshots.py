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
Snapshots manager for PyTorch.
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import gin
import numpy as np
import pandas as pd
import torch as t

@gin.configurable()
class SnapshotManager:
    """
    PyTorch Snapshot Manager.
    Only one, the "latest", state is supported.
    """
    def __init__(self,
                 snapshot_dir: str,
                 total_iterations: int,
                 logging_frequency: int = 100,
                 snapshot_frequency: int = 1000):
        self.model_snapshot_file = os.path.join(snapshot_dir, 'model')
        self.optimizer_snapshot_file = os.path.join(snapshot_dir, 'optimizer')
        self.losses_file = os.path.join(snapshot_dir, 'losses')
        self.iteration_file = os.path.join(snapshot_dir, 'iteration')
        self.time_tracking_file = os.path.join(snapshot_dir, 'time')
        self.logging_frequency = max(1, min(logging_frequency, total_iterations // 3))
        self.snapshot_frequency = max(1, min(snapshot_frequency, total_iterations))
        self.start_time = None
        self.losses = {'training': {}, 'validation': {}}
        self.time_track = {}

    def restore(self, model: Optional[t.nn.Module], optimizer: Optional[t.optim.Optimizer]) -> int:
        """
        Restore a model and optimizer, by mutating their state, and return the iteration number on which
        the state was persisted.

        The losses and model/optimizer state snapshots have different frequencies, thus any losses which were
        registered after the latest model state snapshot will be erased during the restoration process.

        :param model: Model architecture, weights of which should be restored.
        :param optimizer: Optimizer instance, parameters of which should be restored.
        :return: Iteration number.
        """
        if model is not None and os.path.isfile(self.model_snapshot_file):
            model.load_state_dict(t.load(self.model_snapshot_file))
        if optimizer is not None and os.path.isfile(self.optimizer_snapshot_file):
            optimizer.load_state_dict(t.load(self.optimizer_snapshot_file))
        iteration = t.load(self.iteration_file)['iteration'] if os.path.isfile(self.iteration_file) else 0
        if os.path.isfile(self.losses_file):
            losses = t.load(self.losses_file)
            # remove the losses logs which were registered after the last state snapshot.
            training_losses = {k: v for k, v in losses['training'].items() if k <= iteration}
            validation_losses = {k: v for k, v in losses['validation'].items() if k <= iteration}
            self.losses = {'training': training_losses, 'validation': validation_losses}
            self.snapshot(self.losses_file, self.losses)
        if os.path.isfile(self.time_tracking_file):
            self.time_track = t.load(self.time_tracking_file)
        return iteration

    def load_training_losses(self) -> pd.DataFrame:
        """
        Load training losses into a dataframe.

        :return: Training losses in pandas DatFrame.
        """
        if os.path.isfile(self.losses_file):
            losses = t.load(self.losses_file)['training']
            return pd.DataFrame(losses, index=[0])[sorted(losses.keys())].T
        else:
            return pd.DataFrame([np.nan])

    def enable_time_tracking(self):
        """
        Enable time tracking to estimate training time.
        """
        self.start_time = time.time()

    def register(self,
                 iteration: int,
                 training_loss: float,
                 validation_loss: float,
                 model: t.nn.Module,
                 optimizer: Optional[t.optim.Optimizer]) -> None:
        """"
        Register an iteration, the snapshot manager keeps tracking of the frequencies of persistence,
        thus this method should be invoked after each iteration.
        """
        if iteration == 1 or iteration % self.logging_frequency == 0:
            self.losses['training'][iteration] = training_loss
            self.losses['validation'][iteration] = validation_loss
            self.snapshot(self.losses_file, self.losses)
        if iteration % self.snapshot_frequency == 0:
            self.snapshot(self.model_snapshot_file, model.state_dict())
            if optimizer is not None:
                self.snapshot(self.optimizer_snapshot_file, optimizer.state_dict())
            self.snapshot(self.iteration_file, {'iteration': iteration})
            if self.start_time is not None:
                self.time_track[iteration] = time.time() - self.start_time
                self.snapshot(self.time_tracking_file, self.time_track)
                self.start_time = time.time()

    @staticmethod
    def snapshot(path: str, data: Dict) -> None:
        """
        Atomic persistence for data dictionary.

        :param path: Where to persist.
        :param data: What to persist.
        """
        dir_path = os.path.dirname(path)
        if not os.path.isdir(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(dir=dir_path, delete=False, mode='wb')
        t.save(data, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        os.rename(temp_file.name, path)
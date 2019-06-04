import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

params_file_name = 'params.json'
ts_indices_file_name = 'ts_ids_subset.npz'
input_mask_file_name = 'input_mask.npz'


class ExperimentParameters(NamedTuple):
    repeat: int
    training_split: str
    input_size: int
    ts_per_model_ratio: float
    input_dropout: float
    batch_size: int
    iterations: int
    loss_name: str
    init_lr: float
    training_checkpoint_interval: int
    model_type: str

    def persist(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self._asdict(), f)

    @staticmethod
    def load(file_path: str) -> 'ExperimentParameters':
        with open(file_path, 'r') as f:
            return ExperimentParameters(**json.load(f))


class M4Experiment(NamedTuple):
    parameters: ExperimentParameters
    timeseries_indices: np.ndarray
    input_mask: np.ndarray

    def persist(self, experiment_dir_path: str) -> None:
        Path(experiment_dir_path).mkdir(parents=True, exist_ok=False)
        self.parameters.persist(os.path.join(experiment_dir_path, params_file_name))
        self.timeseries_indices.dump(os.path.join(experiment_dir_path, ts_indices_file_name))
        self.input_mask.dump(os.path.join(experiment_dir_path, input_mask_file_name))

    @staticmethod
    def load(experiment_dir_path: str) -> 'M4Experiment':
        return M4Experiment(parameters=ExperimentParameters.load(os.path.join(experiment_dir_path, params_file_name)),
                            timeseries_indices=np.load(os.path.join(experiment_dir_path, ts_indices_file_name),
                                                       allow_pickle=True),
                            input_mask=np.load(os.path.join(experiment_dir_path, input_mask_file_name),
                                               allow_pickle=True))

import json
import os
from typing import NamedTuple

params_file_name = 'params.json'


class Experiment(NamedTuple):
    dataset_split: str
    training_checkpoint_interval: int
    input_horizons: int
    ts_per_model: float
    input_dropout: float
    loss_name: str
    batch_size: int
    iterations: int
    init_lr: float
    model_type: str

    def persist(self, experiment_dir_path: str) -> None:
        with open(os.path.join(experiment_dir_path, params_file_name), 'w') as f:
            json.dump(self._asdict(), f)

    @staticmethod
    def load(experiment_dir_path: str) -> 'Experiment':
        with open(os.path.join(experiment_dir_path, params_file_name), 'r') as f:
            return Experiment(**json.load(f))

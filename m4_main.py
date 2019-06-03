import argparse
import os
import time
from pathlib import Path

from m4.dataset import M4Dataset, M4DatasetSplit
from m4.experiment import Experiment
from m4.settings import DEFAULT_EXPERIMENTS_DIR

training_parameters = {
    'dataset_split': 'train',
    'training_checkpoint_interval': 1000,
    'ts_per_model': 0.2,
    'input_dropout': 0.25,
    'batch_size': 1024,
    'iterations': 30001,
    'init_lr': 0.001
}


def load_training_dataset():
    M4Dataset(split=M4DatasetSplit.TRAIN)
    M4Dataset(split=M4DatasetSplit.TRAIN_SUBSET)
    M4Dataset(split=M4DatasetSplit.VALIDATION_SUBSET)


def init_ensembles(name: str = ''):
    timestamp = time.strftime('%y%m%d_%H%M%S')

    for loss_name in ['MASE', 'MAPE', 'SMAPE']:
        for input_horizons in range(2, 8):
            for repeat in range(0, 10):
                dir_name = f'loss={loss_name},input_h={input_horizons},repeat={repeat}'
                generic_experiment_path = os.path.join(DEFAULT_EXPERIMENTS_DIR, f'{timestamp}_generic_{name}', dir_name)
                interpretable_experiment_path = os.path.join(DEFAULT_EXPERIMENTS_DIR,
                                                             f'{timestamp}_interpretable_{name}', dir_name)
                Path(generic_experiment_path).mkdir(parents=True, exist_ok=False)
                Path(interpretable_experiment_path).mkdir(parents=True, exist_ok=False)

                Experiment(model_type='generic',
                           loss_name=loss_name,
                           input_horizons=input_horizons,
                           **training_parameters).persist(generic_experiment_path)

                Experiment(model_type='interpretable',
                           loss_name=loss_name,
                           input_horizons=input_horizons,
                           **training_parameters).persist(interpretable_experiment_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', metavar='CMD', type=str, choices=['load_training_dataset'],
                        help='Command to execute')

    args = parser.parse_args()

    if args.cmd == 'load_training_dataset':
        load_training_dataset()

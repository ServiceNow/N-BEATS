import argparse
import os
import time

from m4.dataset import M4Dataset, M4DatasetSplit
from m4.experiment import ExperimentParameters, M4Experiment
from m4.settings import M4_EXPERIMENTS_DIR
from m4.utils import build_experiment_name, params_cartesian_product, build_input_mask

training_parameters = {
    'repeat': list(range(10)),  # must always be an array even for only one repeat, for example: [1]

    # Training Dataset
    'training_split': 'train',
    'input_size': list(range(2, 8)),
    'ts_per_model_ratio': 0.2,
    'input_dropout': 0.25,
    'batch_size': 1024,
    'iterations': 30001,

    # Loss
    'loss_name': ['MASE', 'MAPE', 'SMAPE'],

    # Optimization
    'init_lr': 0.001,
    'training_checkpoint_interval': 1000,

    # Model architecture
    'model_type': 'generic'
}


def load_training_dataset():
    """
    Load training datasets, it will download and cache dataset on the first run.

    :return:
    """
    M4Dataset(split=M4DatasetSplit.TRAIN)
    M4Dataset(split=M4DatasetSplit.TRAIN_SUBSET)
    M4Dataset(split=M4DatasetSplit.VALIDATION_SUBSET)


def init_experiment(name: str = ''):
    if name == '':
        name = training_parameters['model_type']
    timestamp = time.strftime(f'%y%m%d_%H%M%S')
    dataset = M4Dataset(split=M4DatasetSplit[training_parameters['training_split'].upper()])
    experiments_dir_path = os.path.join(M4_EXPERIMENTS_DIR, f'{timestamp}_{name}')
    for i, parameters_instance in enumerate(params_cartesian_product(training_parameters)):
        experiment_parameters = ExperimentParameters(**training_parameters, **parameters_instance)
        M4Experiment(parameters=experiment_parameters,
                     timeseries_indices=dataset.sample_indices(experiment_parameters.ts_per_model_ratio),
                     input_mask=build_input_mask(experiment_parameters.input_dropout)).\
            persist(os.path.join(experiments_dir_path, build_experiment_name(parameters_instance)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', metavar='CMD', type=str, choices=['load_training_dataset', 'init_experiment'],
                        help='Command to execute')
    parser.add_argument('--name', type=str, default='', help='Experiment name')

    args = parser.parse_args()

    if args.cmd == 'load_training_dataset':
        load_training_dataset()
    elif args.cmd == 'init_experiment':
        init_experiment(args.name)

import argparse
import logging
import os
import time
import tensorflow as tf

from tqdm import tqdm

from m4.dataset import M4Dataset, M4DatasetSplit
from m4.experiment import ExperimentParameters, M4Experiment
from m4.model import train, predict
from m4.settings import M4_EXPERIMENTS_DIR
from m4.utils import build_experiment_name, params_cartesian_product, build_input_mask

training_parameters = {
    'repeat': list(range(10)),  # must always be an array even for only one repeat, for example: [1]

    # training dataset
    'training_split': 'train',
    'input_size': list(range(2, 8)),
    'ts_per_model_ratio': 0.2,
    'input_dropout': 0.25,
    'batch_size': 1024,

    # training parameters
    'loss_name': ['MASE', 'MAPE', 'SMAPE'],
    'init_lr': 0.001,
    'weight_decay': 0.0,
    'iterations': 30001,

    'training_checkpoint_interval': 10,

    # model architecture
    'model_type': 'generic',  # or 'interpretable'

    # generic model parameters (these parameters will be ignored for 'interpretable' model type)
    'stacks': 30,
    'blocks_in_stack': 1,
    'block_fc_size': 512,
    'block_fc_layers': 4,

    # interpretable model parameters (these parameters will be ignored for 'generic' model type)
    'trend_blocks': 3,
    'trend_block_fc_size': 256,
    'trend_block_fc_layers': 4,
    'trend_order': 3,
    'seasonality_blocks': 3,
    'seasonality_block_fc_size': 2048,
    'seasonality_block_fc_layers': 4,
    'seasonality_num_harmonics': 1
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
    for parameters_instance in tqdm(params_cartesian_product(training_parameters)):
        experiment_parameters = ExperimentParameters(**{**training_parameters, **parameters_instance})
        M4Experiment(parameters=experiment_parameters,
                     timeseries_indices=dataset.sample_indices(experiment_parameters.ts_per_model_ratio),
                     input_mask=build_input_mask(experiment_parameters.input_dropout)).\
            persist(os.path.join(experiments_dir_path, build_experiment_name(parameters_instance)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', metavar='CMD', type=str, choices=['load_training_dataset',
                                                                 'init_experiment', 'train', 'predict'],
                        help='Command to execute')
    parser.add_argument('--name', type=str, default='', help='Experiment name')

    args = parser.parse_args()

    if args.cmd == 'load_training_dataset':
        load_training_dataset()
    elif args.cmd == 'init_experiment':
        init_experiment(args.name)
    elif args.cmd == 'train':
        experiment_path = os.path.join(M4_EXPERIMENTS_DIR, args.name)
        log_path = os.path.join(experiment_path, 'experiment.log')
        tf.logging.log(msg=f'redirect logging to {log_path}')
        logging.basicConfig(filename=log_path, filemode='w')
        train(experiment_path)
        predict(experiment_path)
    elif args.cmd == 'predict':
        predict(os.path.join(M4_EXPERIMENTS_DIR, args.name))

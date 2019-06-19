import argparse
import os
import time

from tqdm import tqdm

from m4.dataset import M4Dataset, M4DatasetSplit
from m4.ensemble import experiment_ensemble
from m4.experiment import ExperimentParameters, M4Experiment
from m4.model import train, predict
from m4.parameters import training_parameters
from m4.settings import M4_EXPERIMENTS_DIR, M4_PREDICTION_FILE_NAME
from m4.summary import summary
from m4.utils import build_experiment_name, params_cartesian_product, build_feature_bagging_mask


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
    experiment_name = f'{timestamp}_{name}'
    experiments_dir_path = os.path.join(M4_EXPERIMENTS_DIR, experiment_name)
    for parameters_instance in tqdm(params_cartesian_product(training_parameters)):
        experiment_parameters = ExperimentParameters(**{**training_parameters, **parameters_instance})
        M4Experiment(parameters=experiment_parameters,
                     timeseries_indices=dataset.sample_indices(experiment_parameters.ts_per_model_ratio),
                     feature_bagging=build_feature_bagging_mask(experiment_parameters.feature_bagging)). \
            persist(os.path.join(experiments_dir_path, build_experiment_name(parameters_instance)))
    print(experiment_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', metavar='CMD', type=str, choices=['download_training_dataset',
                                                                 'init_experiment', 'train', 'summary'],
                        help='Command to execute')
    parser.add_argument('--experiment', type=str, default='', help='Experiment name')
    parser.add_argument('--model', type=str, default='', help='Model name')
    parser.add_argument('--validation', type=bool, default=False, help='Validation mode')

    args = parser.parse_args()

    if args.cmd == 'download_training_dataset':
        load_training_dataset()
    elif args.cmd == 'init_experiment':
        init_experiment(args.experiment)
    elif args.cmd == 'train':
        experiment_path = os.path.join(M4_EXPERIMENTS_DIR, args.experiment, args.model)
        train(experiment_path)
        predict(experiment_path)
    elif args.cmd == 'summary':
        predictions_file_path = os.path.join(M4_EXPERIMENTS_DIR, args.experiment, M4_PREDICTION_FILE_NAME)
        if not os.path.isfile(predictions_file_path):
            experiment_ensemble(experiment_dir=os.path.join(M4_EXPERIMENTS_DIR, args.experiment), overwrite=False)
        result = summary(prediction_csv_path=predictions_file_path,
                         training_set=M4Dataset(
                             M4DatasetSplit.TRAIN_SUBSET if args.validation else M4DatasetSplit.TRAIN),
                         test_set=M4Dataset(
                             M4DatasetSplit.VALIDATION_SUBSET if args.validation else M4DatasetSplit.TEST))
        print(result)

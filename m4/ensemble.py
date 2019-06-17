import logging
import os
from glob import glob

import pandas as pd
from tqdm import tqdm

from m4.settings import M4_PREDICTION_FILE_NAME
from m4.utils import csv_to_df


def ensemble(result_path: str, *predictions: str) -> None:
    """
    Compute predictions median values.

    :param result_path: Where to persist the result.
    :param predictions: Predictions to compute median from.
    :return: None
    """

    if os.path.isfile(result_path):
        raise Exception(f'{result_path} already exists.')

    if len(predictions) < 1:
        raise Exception('ensemble needs at least one prediction.')

    logging.debug(f'Averaging {len(predictions)} predictions.')

    results_dict = {i: df for i, df in enumerate(map(lambda p: csv_to_df(p, id_column_index=0), tqdm(predictions)))}
    panel = pd.Panel(results_dict)
    result = panel.median(axis=0)
    result.columns = [f'F{i}' for i in range(1, len(result.columns) + 1)]
    result.to_csv(result_path)


def experiment_ensemble(experiment_dir: str,
                        experiment_filter: str = '*',
                        overwrite: bool = False,
                        target_file: str = M4_PREDICTION_FILE_NAME,
                        result_file: str = M4_PREDICTION_FILE_NAME) -> None:
    """
    Build median prediction from all predictions in the given experiments directory.
    The result prediction will be written to experiments directory.

    :param experiment_dir: Experiment directory which contains sub-experiments.
    :param experiment_filter: Experiments filter.
    :param overwrite: Overwrite existing result if True.
    :param target_file: File name to search for.
    :param result_file: Result file name.
    :return: None
    """
    result_path = os.path.join(experiment_dir, result_file)

    if os.path.isfile(result_path):
        if overwrite:
            os.remove(result_path)
        else:
            raise Exception(f'Ensemble result is already in the experiment directory, use overwrite flag to overwrite.')

    ensemble(result_path, *glob(f'{experiment_dir}/{experiment_filter}/{target_file}'))

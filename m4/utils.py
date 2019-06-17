import logging
import os
import pathlib
import sys
import urllib
from itertools import product
from typing import Optional
from urllib import request

import numpy as np
import pandas as pd
import tensorflow as tf

from m4.settings import M4_INPUT_MAXSIZE


def csv_to_df(csv_file_path: str, id_column_index: Optional[int] = None) -> pd.DataFrame:
    """
    Convert csv to pandas dataframe, if id column specified it will be used as dataframe index.

    :param csv_file_path: path to csv file.
    :param id_column_index: id column index.
    :return: Pandas DataFrame.
    """
    df = pd.read_csv(csv_file_path)
    if id_column_index is not None:
        df.set_index(df.columns[id_column_index], inplace=True)
    return df


def url_file_name(url: str) -> str:
    """
    Extract file name from url (last part of the url).

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1]


def download_url(url: str, target_directory: str) -> None:
    """
    Download a file to the given target directory.

    :param url: URL to download
    :param target_directory: Directory to store the file to.
    """
    file_name = url_file_name(url)
    file_path = os.path.join(target_directory, file_name)

    def progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading {} {} {:.1f}%'.format(file_path,
                                                              url,
                                                              float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        pathlib.Path(target_directory).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {file_name} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_name} {file_info.st_size} bytes.')


def build_input_dropout_mask(dropout_ratio: float):
    indices = np.arange(M4_INPUT_MAXSIZE, dtype=np.int32)
    sampled_indices = np.random.choice(indices, size=int(M4_INPUT_MAXSIZE * (1 - dropout_ratio)), replace=False)
    input_mask = np.zeros(M4_INPUT_MAXSIZE, dtype=np.float32)
    input_mask[sampled_indices] = 1.0

    # Make sure that the last sample in insample is always part of the model, for it provides most information
    input_mask[0] = 1.0
    return input_mask


def summary_log(log_dir, writer=None):
    """Convenient wrapper for writing summaries."""
    if writer is None:
        writer = tf.summary.FileWriter(log_dir)

    def call(step, **value_dict):
        summary = tf.Summary()
        for tag, value in value_dict.items():
            summary.value.add(tag=tag, simple_value=value)
        writer.add_summary(summary, step)
        writer.flush()

    return call


def params_cartesian_product(param_dict):
    """
    Find the variables in param_dict and yields every instance part of the cartesian product.

    :param param_dict: dictionary of parameters. Every item that is a list will be cross-validated.
    :return: A dictionary of parameters where lists are replaced with one of their instance.
    """
    variables = []
    for key, val in param_dict.items():
        if isinstance(val, list):
            variables.append([(key, element) for element in val])

    for experiment in product(*variables):
        yield dict(experiment)


def build_experiment_name(parameters):
    """
    Create a readable name containing the name and value of the experiment parameters.

    :param parameters: Dictionary of experiment parameters.
    """
    args = []
    for name, value in parameters.items():
        if isinstance(value, float):
            args.append("%s=%.4g" % (name, value))
        else:
            args.append("%s=%s" % (name, value))
    return ','.join(args)


def get_masep(insample, freq):
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))
    return masep

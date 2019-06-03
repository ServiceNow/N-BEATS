import logging
import os
import pathlib
import sys
import urllib
from typing import Optional
from urllib import request

import pandas as pd
import tensorflow as tf


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
        sys.stdout.write('\rDownloading {} {:.1f}%'.format(file_path,
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

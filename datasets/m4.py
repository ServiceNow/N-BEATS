"""
M4 Dataset
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

TRAINING_DATASET_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2017/12/M4DataSet.zip'
TEST_DATASET_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2018/07/M-test-set.zip'
INFO_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2018/12/M4Info.csv'

DATASET_PATH = os.path.join(DATASETS_PATH, 'm4')

TRAINING_DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(TRAINING_DATASET_URL))
TEST_DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(TEST_DATASET_URL))
INFO_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(INFO_URL))

TRAINING_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'training.npz')
TEST_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'test.npz')


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True) -> 'M4Dataset':
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        m4_info = pd.read_csv(INFO_FILE_PATH)
        return M4Dataset(ids=m4_info.M4id.values,
                         groups=m4_info.SP.values,
                         frequencies=m4_info.Frequency.values,
                         horizons=m4_info.Horizon.values,
                         values=np.load(
                             TRAINING_DATASET_CACHE_FILE_PATH if training else TEST_DATASET_CACHE_FILE_PATH,
                             allow_pickle=True))

    @staticmethod
    def download() -> None:
        """
        Download M4 dataset if doesn't exist.
        """
        if os.path.isdir(DATASET_PATH):
            logging.info(f'skip: {DATASET_PATH} directory already exists.')
            return

        # Download and load m4 info file.
        download(INFO_URL, INFO_FILE_PATH)
        m4_ids = pd.read_csv(INFO_FILE_PATH).M4id.values

        # Download and cache test set
        download(TRAINING_DATASET_URL, TRAINING_DATASET_FILE_PATH)

        def build_cache(files: str, cache_path: str) -> None:
            timeseries_dict = OrderedDict(list(zip(m4_ids, [[]] * len(m4_ids))))
            logging.info(f'Caching {files}')
            patoolib.extract_archive(TRAINING_DATASET_FILE_PATH, outdir=DATASET_PATH)
            for train_csv in tqdm(glob(os.path.join(DATASET_PATH, files))):
                dataset = pd.read_csv(train_csv)
                dataset.set_index(dataset.columns[0], inplace=True)
                for m4id, row in dataset.iterrows():
                    values = row.values
                    timeseries_dict[m4id] = values[~np.isnan(values)]
            np.array(list(timeseries_dict.values())).dump(cache_path)

        build_cache('*-train.csv', TRAINING_DATASET_CACHE_FILE_PATH)
        build_cache('*-test.csv', TEST_DATASET_CACHE_FILE_PATH)


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }

def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)

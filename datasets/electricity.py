# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Electricity Dataset
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import fire
import numpy as np
import patoolib
from tqdm import tqdm

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

"""
Hourly aggregated dataset from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

As it is used in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
Dataset was also compared with the one built by the TRMF paper's author:
https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
"""

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

DATASET_DIR = os.path.join(DATASETS_PATH, 'electricity')
DATASET_FILE_PATH = os.path.join(DATASET_DIR, url_file_name(DATASET_URL))
RAW_DATA_FILE_PATH = os.path.join(DATASET_DIR, 'LD2011_2014.txt')

CACHE_FILE_PATH = os.path.join(DATASET_DIR, 'electricity.npz')
DATES_CACHE_FILE_PATH = os.path.join(DATASET_DIR, 'dates.npz')

@dataclass()
class ElectricityMeta:
    horizon = 24
    clients = 370
    time_steps = 26304
    seasonal_pattern = 'Hourly'
    frequency = 24

@dataclass()
class ElectricityDataset:
    ids: np.ndarray
    values: np.ndarray
    dates: np.ndarray

    @staticmethod
    def load() -> 'ElectricityDataset':
        """
        Load Electricity dataset from cache.
        """
        value = np.load(CACHE_FILE_PATH, allow_pickle=True)
        return ElectricityDataset(
            ids=np.array(list(range(len(value)))),
            values=np.load(CACHE_FILE_PATH, allow_pickle=True),
            dates=np.load(DATES_CACHE_FILE_PATH, allow_pickle=True))

    def split_by_date(self, cut_date: str, include_cut_date: bool = True) -> Tuple['ElectricityDataset', 'ElectricityDataset']:
        """
        Split dataset by date.

        :param cut_date: Cut date in "%Y-%m-%d %H" format
        :param include_cut_date: Include cut_date in the split if true, not otherwise.
        :return: Two parts of dataset: the left part contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        date = datetime.strptime(cut_date, '%Y-%m-%d %H')
        left_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
            if record_date < date or (include_cut_date and record_date == date):
                left_indices.append(i)
            else:
                right_indices.append(i)
        return ElectricityDataset(ids=self.ids,
                                  values=self.values[:, left_indices],
                                  dates=self.dates[left_indices]), \
               ElectricityDataset(ids=self.ids,
                                  values=self.values[:, right_indices],
                                  dates=self.dates[right_indices])

    def split(self, cut_point: int) -> Tuple['ElectricityDataset', 'ElectricityDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: left contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        return ElectricityDataset(ids=self.ids,
                                  values=self.values[:, :cut_point],
                                  dates=self.dates[:cut_point]), \
               ElectricityDataset(ids=self.ids,
                                  values=self.values[:, cut_point:],
                                  dates=self.dates[cut_point:])

    def time_points(self):
        return self.dates.shape[0]

    @staticmethod
    def download():
        """
        Download Electricity dataset.
        """
        if os.path.isdir(DATASET_DIR):
            logging.info(f'skip: {DATASET_DIR} directory already exists.')
            return
        download(DATASET_URL, DATASET_FILE_PATH)
        patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_DIR)
        with open(RAW_DATA_FILE_PATH, 'r') as f:
            raw = f.readlines()

        # based on data downloaded by script:
        # https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
        # the first year of data was ignored.
        # The raw data frequency is 15 minutes, thus we ignore header, first record, and 4 * 24 * 365 data points
        header = 1
        ignored_first_values = header + (365 * 24 * 4)
        parsed_values = list(map(lambda raw_line: raw_line.replace(',', '.').strip().split(';')[1:],
                                 raw[ignored_first_values:]))
        data = np.array(parsed_values).astype(np.float)

        # aggregate to hourly
        aggregated = []
        for i in tqdm(range(0, data.shape[0], 4)):
            aggregated.append(data[i:i + 4, :].sum(axis=0))
        aggregated = np.array(aggregated)

        dataset = aggregated.T  # use time step as second dimension.
        logging.info(f'Caching matrix {dataset.shape} to {CACHE_FILE_PATH}')
        dataset.dump(CACHE_FILE_PATH)
        logging.info(f'Caching dates to {DATES_CACHE_FILE_PATH}')
        dates = list(map(lambda raw_line: raw_line.replace(',', '.').strip().split(';')[0], raw[ignored_first_values:]))
        # ignore first hour, for its values are aggregated to the next hour.
        np.unique(list(
            map(lambda s: datetime.strptime(s[1:-1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H'), dates)))[1:].dump(
            DATES_CACHE_FILE_PATH)

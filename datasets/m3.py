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
M3 Dataset
"""
import logging
import os
from dataclasses import dataclass

import fire
import numpy as np
import pandas as pd

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

DATASET_URL = 'https://forecasters.org/data/m3comp/M3C.xls'
FORECASTS_URL = 'https://forecasters.org/data/m3comp/M3Forecast.xls'

DATASET_PATH = os.path.join(DATASETS_PATH, 'm3')
DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))

TRAINING_SET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'training.npy')
TEST_SET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'test.npy')
IDS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'ids.npy')
GROUPS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'groups.npy')
HORIZONS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'horizons.npy')


@dataclass()
class M3Meta:
    seasonal_patterns = ['M3Year', 'M3Quart', 'M3Month', 'M3Other']
    horizons = [6, 8, 18, 8]
    frequency = [1, 4, 12, 1]
    horizons_map = {
        'M3Year': 6,
        'M3Quart': 8,
        'M3Month': 18,
        'M3Other': 8
    }
    frequency_map = {
        'M3Year': 1,
        'M3Quart': 4,
        'M3Month': 12,
        'M3Other': 1
    }


@dataclass()
class M3Dataset:
    ids: np.ndarray
    groups: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True) -> 'M3Dataset':
        values_file = TRAINING_SET_CACHE_FILE_PATH if training else TEST_SET_CACHE_FILE_PATH
        return M3Dataset(ids=np.load(IDS_CACHE_FILE_PATH, allow_pickle=True),
                         groups=np.load(GROUPS_CACHE_FILE_PATH, allow_pickle=True),
                         horizons=np.load(HORIZONS_CACHE_FILE_PATH, allow_pickle=True),
                         values=np.load(values_file, allow_pickle=True))

    def to_training_subset(self):
        return M3Dataset(ids=self.ids,
                         groups=self.groups,
                         horizons=self.horizons,
                         values=np.array([v[:-self.horizons[i]] for i, v in enumerate(self.values)]))

    def to_hp_search_training_subset(self):
        return M3Dataset(ids=self.ids,
                         groups=self.groups,
                         horizons=self.horizons,
                         values=np.array([v[:-2 * self.horizons[i]] for i, v in enumerate(self.values)]))

    @staticmethod
    def download() -> None:
        """
        Download M3 dataset if doesn't exist.
        """
        if os.path.isdir(DATASET_PATH):
            logging.info(f'skip: {DATASET_PATH} directory already exists.')
            return

        download(DATASET_URL, DATASET_FILE_PATH)
        logging.info('Load and cache forecasts ...')

        ids = []
        groups = []
        horizons = []
        training_values = []
        test_values = []

        for sp in M3Meta.seasonal_patterns:
            horizon = M3Meta.horizons_map[sp]
            dataset = pd.read_excel(DATASET_FILE_PATH, sheet_name=sp)
            ids.extend(dataset[['Series']].values[:, 0])
            horizons.extend(dataset['NF'].values)
            groups.extend(np.array([sp] * len(dataset)))
            training_values.extend([ts[~np.isnan(ts)][:-horizon] for ts in dataset[dataset.columns[6:]].values])
            test_values.extend([ts[~np.isnan(ts)][-horizon:] for ts in dataset[dataset.columns[6:]].values])

        np.save(IDS_CACHE_FILE_PATH, ids, allow_pickle=True)
        np.save(GROUPS_CACHE_FILE_PATH, groups, allow_pickle=True)
        np.save(HORIZONS_CACHE_FILE_PATH, horizons, allow_pickle=True)
        np.save(TRAINING_SET_CACHE_FILE_PATH, training_values, allow_pickle=True)
        np.save(TEST_SET_CACHE_FILE_PATH, test_values, allow_pickle=True)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    fire.Fire()

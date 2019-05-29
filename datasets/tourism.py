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
Tourism Dataset
"""
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patoolib

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

DATASET_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'

DATASET_PATH = os.path.join(DATASETS_PATH, 'tourism')
DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))


@dataclass()
class TourismMeta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly']
    horizons = [4, 8, 24]
    frequency = [1, 4, 12]
    horizons_map = {
        'Yearly': 4,
        'Quarterly': 8,
        'Monthly': 24
    }
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12
    }


@dataclass()
class TourismDataset:
    ids: np.ndarray
    groups: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True) -> 'TourismDataset':
        """
        Load Tourism dataset from cache.

        :param training: Load training part if training is True, test part otherwise.
        """
        ids = []
        groups = []
        horizons = []
        values = []

        for group in TourismMeta.seasonal_patterns:

            train = pd.read_csv(os.path.join(DATASET_PATH, f'{group.lower()}_in.csv'),
                                header=0, delimiter=",")
            test = pd.read_csv(os.path.join(DATASET_PATH, f'{group.lower()}_oos.csv'),
                               header=0, delimiter=",")

            horizons.extend(list(test.iloc[0].astype(int)))
            groups.extend([group] * len(train.columns))

            if group == 'Yearly':
                train_meta = train[:2]
                meta_length = train_meta.iloc[0].astype(int)
                test = test[2:].reset_index(drop=True).T
                train = train[2:].reset_index(drop=True).T
            else:
                train_meta = train[:3]
                meta_length = train_meta.iloc[0].astype(int)
                test = test[3:].reset_index(drop=True).T
                train = train[3:].reset_index(drop=True).T

            ids.extend(list(train.index))

            if training:
                dataset = train
            else:
                dataset = test

            values.extend([ts[:ts_length] for ts, ts_length in zip(dataset.values, meta_length)])

        return TourismDataset(ids=np.array(ids),
                              groups=np.array(groups),
                              horizons=np.array(horizons),
                              values=np.array(values))

    @staticmethod
    def download():
        """
        Download Tourism dataset.
        """
        if os.path.isdir(DATASET_PATH):
            logging.info(f'skip: {DATASET_PATH} directory already exists.')
            return
        download(DATASET_URL, DATASET_FILE_PATH)
        patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_PATH)

    def to_hp_search_training_subset(self):
        return TourismDataset(ids=self.ids,
                              groups=self.groups,
                              horizons=self.horizons,
                              values=np.array([v[:-self.horizons[i]] for i, v in enumerate(self.values)]))

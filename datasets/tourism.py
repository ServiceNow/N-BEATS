"""
Tourism Dataset
"""
import logging
import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import patoolib

from common.http_utils import download, url_file_name
from common.settings import DATASETS_DIR

DATASET_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'

DATASET_DIR = os.path.join(DATASETS_DIR, 'tourism')
DATASET_FILE_PATH = os.path.join(DATASET_DIR, url_file_name(DATASET_URL))


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
class TourismDataset(NamedTuple):
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

            train = pd.read_csv(os.path.join(DATASET_DIR, f'{group.lower()}_in.csv'),
                                header=0, delimiter=",")
            test = pd.read_csv(os.path.join(DATASET_DIR, f'{group.lower()}_oos.csv'),
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
        if os.path.isdir(DATASET_DIR):
            logging.info(f'skip: {DATASET_DIR} directory already exists.')
            return
        download(DATASET_URL, DATASET_FILE_PATH)
        patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_DIR)

    def to_hp_search_training_subset(self):
        return TourismDataset(ids=self.ids,
                              groups=self.groups,
                              horizons=self.horizons,
                              values=np.array([v[:-self.horizons[i]] for i, v in enumerate(self.values)]))

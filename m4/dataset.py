import logging
import os
from collections import OrderedDict
from enum import Enum
from glob import glob
from typing import Dict, List, NamedTuple

import numpy as np
import patoolib
from tqdm import tqdm

from m4.settings import M4_DATA_DIR, M4_INFO_URL, M4_TEST_SET_URL, M4_TRAINING_SET_URL
from m4.utils import csv_to_df, download_url, url_file_name


class M4Info:
    def __init__(self, data_dir_path: str = M4_DATA_DIR):
        info_file_name = url_file_name(M4_INFO_URL)
        meta_file_path = os.path.join(data_dir_path, info_file_name)
        if not os.path.isfile(meta_file_path):
            download_url(M4_INFO_URL, data_dir_path)
        self.data = csv_to_df(meta_file_path, id_column_index=0)
        self.total_number_of_timeseries = len(self.data)
        self.ids = self.data.index.values
        self.horizons = self.data.Horizon.drop_duplicates().values


class M4DatasetSplit(Enum):
    TRAIN = 'train'
    TEST = 'test'
    TRAIN_SUBSET = 'train-subset'
    VALIDATION_SUBSET = 'validation-subset'


class M4Batch(NamedTuple):
    horizon: int
    inputs: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    masep: np.ndarray


class M4Dataset:
    def __init__(self,
                 split: M4DatasetSplit = M4DatasetSplit.TRAIN):
        self.split = split
        self.info = M4Info(M4_DATA_DIR)
        self.data = self.__get_cached_dataset(os.path.join(M4_DATA_DIR, f'{self.split.value}_dataset.npz'))

    def sample_indices(self, ratio: float) -> np.ndarray:
        """
        Sample time series (by index in M4Info.csv).

        :param ratio: ratio of the total number of time series.
        :return: Collection of sampled indices.
        """
        indices = np.arange(len(self.info.ids), dtype=np.int32)
        return np.random.choice(indices, size=int(ratio * len(self.info.ids)), replace=False)

    def next_batch(self, batch_size: int = 64) -> M4Batch:
        """
        Sample a batch from the dataset.

        :param batch_size: Number of timeseries in the batch.
        :return: Batch dictionary
        """
        # TODO: implement next_batch
        return M4Batch(horizon=0)

    def __get_cached_dataset(self, cache_file_path: str) -> np.ndarray:
        if not os.path.isfile(cache_file_path):
            #
            # Download extract and cache.
            #
            dataset_url = M4_TEST_SET_URL if self.split == M4DatasetSplit.TEST else M4_TRAINING_SET_URL
            dataset_name = url_file_name(dataset_url)
            dataset_path = os.path.join(M4_DATA_DIR, dataset_name)
            if not os.path.isfile(dataset_path):
                download_url(dataset_url, M4_DATA_DIR)
                patoolib.extract_archive(dataset_path, outdir=M4_DATA_DIR)
            dataset_subset = 'test' if self.split == M4DatasetSplit.TEST else 'train'

            logging.info(f'Build and cache {self.split.name} dataset')
            # timeseries dictionary will preserve timeseries in the order defined in M4Info.csv
            timeseries_dict = OrderedDict(list(zip(self.info.ids, [[]] * len(self.info.ids))))
            dataset_file_paths = glob(os.path.join(M4_DATA_DIR, f'*-{dataset_subset}.csv'))
            for f in tqdm(dataset_file_paths):
                dataset = csv_to_df(f, id_column_index=0)
                horizons = self.info.data.loc[dataset.index].Horizon.drop_duplicates()
                if horizons.size != 1:
                    raise Exception(f'Loaded dataset does not contain timeseries of same horizon. Dataset path: {f}')
                horizon = horizons.values[0]
                subset = slice(None, None)
                if self.split == M4DatasetSplit.TRAIN_SUBSET:
                    subset = slice(None, -horizon)
                elif self.split == M4DatasetSplit.VALIDATION_SUBSET:
                    subset = slice(-horizon, None)

                for m4id, row in dataset.iterrows():
                    values = row.values
                    timeseries_dict[m4id] = values[~np.isnan(values)][subset]
            np.array(list(timeseries_dict.values())).dump(cache_file_path)
        return np.load(cache_file_path, allow_pickle=True)

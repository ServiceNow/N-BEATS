import logging
import os
import random
from collections import OrderedDict
from enum import Enum
from glob import glob
from typing import NamedTuple, Optional

import numpy as np
import patoolib
from tqdm import tqdm

from m4.settings import M4_DATA_DIR, M4_INFO_URL, M4_INPUT_MAXSIZE, M4_SAMPLING_WINDOW_LIMIT, M4_TEST_SET_URL, \
    M4_TRAINING_SET_URL
from m4.utils import csv_to_df, download_url, get_masep, url_file_name


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
        # TODO: clarify that this is not really MASE for sampled data (although it plays a role of scaler only).
        self.masep = self.__get_cached_masep(os.path.join(M4_DATA_DIR, f'{self.split.value}_masep.npz'))

    def sample_indices(self, ratio: float) -> np.ndarray:
        """
        Sample time series (by index in M4Info.csv).

        :param ratio: ratio of the total number of time series.
        :return: Collection of sampled indices.
        """
        indices = np.arange(len(self.info.ids), dtype=np.int32)
        return np.random.choice(indices, size=int(ratio * len(self.info.ids)), replace=False)

    def sample_batch(self, batch_size: int = 64, indices_filter: Optional[np.ndarray] = None) -> M4Batch:
        """
        Sample a batch from the dataset.

        :param batch_size: Number of timeseries in the batch.
        :param indices_filter:
        :return: Batch dictionary
        """
        horizon_fractions = dict(self.info.data.Horizon.value_counts() / self.info.total_number_of_timeseries)
        sampled_horizon = np.int32(np.random.choice(list(horizon_fractions.keys()), 1,
                                                    p=list(horizon_fractions.values()))[0])
        timeseries_with_same_horizon = np.arange(self.info.total_number_of_timeseries)[
            self.info.data.Horizon == sampled_horizon]

        if indices_filter is None:
            indices_filter = np.arange(self.info.total_number_of_timeseries)

        indices_subset = list(set(indices_filter).intersection(set(timeseries_with_same_horizon)))

        batch_input = np.zeros(shape=(batch_size, M4_INPUT_MAXSIZE), dtype=np.float32)
        batch_target = np.zeros(shape=(batch_size, sampled_horizon), dtype=np.float32)
        batch_target_mask = np.zeros(shape=(batch_size, sampled_horizon), dtype=np.float32)
        history_limit = M4_SAMPLING_WINDOW_LIMIT[sampled_horizon]
        sampled_indices = np.random.choice(indices_subset, size=batch_size, replace=True)
        for i, ts_index in enumerate(sampled_indices):
            ts = self.data[ts_index]
            sampling_limit = max(1, len(ts) - int(history_limit * sampled_horizon))
            sampled_point = random.randint(sampling_limit, len(ts) - 1)
            ts_input = np.flip(ts[max(0, sampled_point - M4_INPUT_MAXSIZE):sampled_point])
            ts_target = ts[sampled_point:min(sampled_point + sampled_horizon, len(ts))]
            batch_input[i, :len(ts_input)] = ts_input
            batch_target[i, :len(ts_target)] = ts_target
            batch_target_mask[i, :len(ts_target)] = 1.0

        return M4Batch(horizon=sampled_horizon,
                       inputs=batch_input,
                       targets=batch_target,
                       target_mask=batch_target_mask,
                       masep=self.masep[sampled_indices])

    def sequential_input_batches(self, batch_size, horizon: int):
        indices = np.arange(self.info.total_number_of_timeseries, dtype=np.int32)
        indices_for_horizon = indices[self.info.data.Horizon == horizon]
        num_batches = int(np.ceil(len(indices_for_horizon) / batch_size))
        for i in range(num_batches):
            batch_indices = indices_for_horizon[i * batch_size: (i + 1) * batch_size]
            batch_timeseries = [self.data[j] for j in batch_indices]
            inputs = np.zeros(shape=(len(batch_timeseries), M4_INPUT_MAXSIZE), dtype=np.float32)
            for ts_index, ts in enumerate(batch_timeseries):
                ts_last_window = np.flip(ts[max(0, len(ts) - M4_INPUT_MAXSIZE):])
                inputs[ts_index, :len(ts_last_window)] = ts_last_window
            yield inputs

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

    def __get_cached_masep(self, cache_file_path: str) -> np.ndarray:
        if not os.path.isfile(cache_file_path):
            masep = np.zeros((len(self.data),), dtype=np.float32)
            for i, ts in enumerate(tqdm(self.data)):
                masep[i] = get_masep(insample=ts, freq=self.info.data.Frequency.values[i])
            masep.dump(cache_file_path)
        return np.load(cache_file_path, allow_pickle=True)

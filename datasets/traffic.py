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
Traffic Dataset
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import patoolib
from tqdm import tqdm

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

"""
Hourly aggregated dataset from https://archive.ics.uci.edu/ml/datasets/PEMS-SF

As it is used in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
Dataset was also compared with the one built by the TRMF paper's author:
https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
"""

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

DATASET_PATH = os.path.join(DATASETS_PATH, 'traffic')
DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))

CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'traffic.npz')
DATES_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'dates.npz')

TRAIN_LABELS_FILE = os.path.join(DATASET_PATH, 'PEMS_trainlabels')
TEST_LABELS_FILE = os.path.join(DATASET_PATH, 'PEMS_testlabels')


@dataclass()
class TrafficMeta:
    horizon = 24
    lanes = 963
    seasonal_pattern = 'Hourly'
    frequency = 24 * 7  # week, same time


@dataclass()
class TrafficDataset:
    ids: np.ndarray
    values: np.ndarray
    dates: np.ndarray

    @staticmethod
    def load() -> 'TrafficDataset':
        """
        Load Traffic dataset from cache.
        :return:
        """
        values = np.load(CACHE_FILE_PATH, allow_pickle=True)
        return TrafficDataset(
            ids=np.array(list(range(len(values)))),
            values=values,
            dates=np.load(DATES_CACHE_FILE_PATH, allow_pickle=True))

    def split_by_date(self, cut_date: str, include_cut_date: bool = True) -> Tuple['TrafficDataset', 'TrafficDataset']:
        """
        Split dataset by date.

        :param cut_date: Cut date in "%Y-%m-%d %H" format
        :param include_cut_date: Include cut_date in the split if true, not otherwise.
        :return: Left and right parts of the split.
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
        return TrafficDataset(ids=self.ids,
                              values=self.values[:, left_indices],
                              dates=self.dates[left_indices]), \
               TrafficDataset(ids=self.ids,
                              values=self.values[:, right_indices],
                              dates=self.dates[right_indices])

    def split(self, cut_point: int) -> Tuple['TrafficDataset', 'TrafficDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: the left part contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        return TrafficDataset(ids=self.ids,
                              values=self.values[:, :cut_point],
                              dates=self.dates[:cut_point]), \
               TrafficDataset(ids=self.ids,
                              values=self.values[:, cut_point:],
                              dates=self.dates[cut_point:])

    def time_points(self):
        return self.dates.shape[0]

    @staticmethod
    def download():
        if os.path.isdir(DATASET_PATH):
            logging.info(f'skip: {DATASET_PATH} directory already exists.')
            return
        download(DATASET_URL, DATASET_FILE_PATH)
        patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_PATH)
        with open(os.path.join(DATASET_PATH, 'PEMS_train'), 'r') as f:
            train_raw_data = f.readlines()
        with open(os.path.join(DATASET_PATH, 'PEMS_test'), 'r') as f:
            test_raw_data = f.readlines()
        with open(os.path.join(DATASET_PATH, 'randperm'), 'r') as f:
            permutations = f.readlines()
        permutations = np.array(permutations[0].rstrip()[1:-1].split(' ')).astype(np.int)

        raw_data = train_raw_data + test_raw_data

        # The assumption below does NOT affect the results, because the splits we use in the publication are
        # based on either dates within the first 6 months, where the labels are aligned or on
        # the last values of dataset. Thus there should not be any confusion with misaligned split points.
        #
        # Dataset dates issue:
        #
        # From the dataset description [https://archive.ics.uci.edu/ml/datasets/PEMS-SF] :
        # "The measurements cover the period from Jan. 1st 2008 to Mar. 30th 2009"
        # and
        # "We remove public holidays from the dataset, as well
        # as two days with anomalies (March 8th 2009 and March 9th 2008)".
        #
        # Based on provided labels, which are days of week, the sequence of days had only 10 gaps by 1 day,
        # where the first 6 correspond to a holiday or anomalous day, but the other 4 gaps happen on "random" dates,
        # meaning we could not find any holiday or the mentioned anomalous days around those dates.
        #
        # More over, the number of days between 2008-01-01 and 2009-03-30 is 455, with only 10 gaps it's
        # not possible to fill dates up to 2009-03-30, it should be 15 gaps (if 2009-01-01 is included, 14 otherwise).
        #
        # Thus, it is not clear if either labels are not correct or the dataset description.
        #
        # Since we are not using any covariates and the split dates after the first 6 months we just fill the gaps with
        # the most common holidays, it does not have any impact on the split points anyway.
        current_date = datetime.strptime('2008-01-01', '%Y-%m-%d')
        excluded_dates = [
            datetime.strptime('2008-01-01', '%Y-%m-%d'),
            datetime.strptime('2008-01-21', '%Y-%m-%d'),
            datetime.strptime('2008-02-18', '%Y-%m-%d'),
            datetime.strptime('2008-03-09', '%Y-%m-%d'),
            datetime.strptime('2008-05-26', '%Y-%m-%d'),
            datetime.strptime('2008-07-04', '%Y-%m-%d'),
            datetime.strptime('2008-09-01', '%Y-%m-%d'),
            datetime.strptime('2008-10-13', '%Y-%m-%d'),
            datetime.strptime('2008-11-11', '%Y-%m-%d'),
            datetime.strptime('2008-11-27', '%Y-%m-%d'),
            datetime.strptime('2008-12-25', '%Y-%m-%d'),
            datetime.strptime('2009-01-01', '%Y-%m-%d'),
            datetime.strptime('2009-01-19', '%Y-%m-%d'),
            datetime.strptime('2009-02-16', '%Y-%m-%d'),
            datetime.strptime('2009-03-08', '%Y-%m-%d'),
        ]
        dates = []
        np_array = []
        for i in tqdm(range(len(permutations))):
            # values
            matrix = raw_data[np.where(permutations == i + 1)[0][0]].rstrip()[1:-1]
            daily = []
            for row_vector in matrix.split(';'):
                daily.append(np.array(row_vector.split(' ')).astype(np.float32))
            daily = np.array(daily)
            if len(np_array) == 0:
                np_array = daily
            else:
                np_array = np.concatenate([np_array, daily], axis=1)

            # dates
            while current_date in excluded_dates:  # skip those in excluded dates
                current_date = current_date + timedelta(days=1)
            dates.extend([(current_date + timedelta(hours=i + 1)).strftime('%Y-%m-%d %H') for i in range(24)])
            current_date = current_date + timedelta(days=1)

        # aggregate 10 minutes events to hourly
        hourly = np.array([list(map(np.mean, zip(*(iter(lane),) * 6))) for lane in tqdm(np_array)])
        logging.info(f'Caching data {hourly.shape} to {CACHE_FILE_PATH}')
        hourly.dump(CACHE_FILE_PATH)
        logging.info(f'Caching dates to {DATES_CACHE_FILE_PATH}')
        np.array(dates).dump(DATES_CACHE_FILE_PATH)

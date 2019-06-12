import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import patoolib

from m4.dataset import M4Dataset, M4Info
from m4.settings import M4_DATA_DIR, M4_NAIVE2_URL
from m4.utils import csv_to_df, download_url, url_file_name


def summary(prediction_csv_path: str, training_set: M4Dataset, test_set: M4Dataset):
    #
    # load Model and Naive2 predictions
    #
    model_prediction = csv_to_df(prediction_csv_path, id_column_index=0).values
    model_prediction = np.array([x[~np.isnan(x)] for x in model_prediction])

    naive2_csv_path = os.path.join(M4_DATA_DIR, 'submission-Naive2.csv')
    if not os.path.isfile(naive2_csv_path):
        naive2_package_path = os.path.join(M4_DATA_DIR, url_file_name(M4_NAIVE2_URL))
        download_url(M4_NAIVE2_URL, target_directory=M4_DATA_DIR)
        patoolib.extract_archive(naive2_package_path, outdir=M4_DATA_DIR)
    naive2_prediction = csv_to_df(naive2_csv_path, id_column_index=0).values
    naive2_prediction = np.array([x[~np.isnan(x)] for x in naive2_prediction])

    #
    # MASE and SMAPE of both Model and Naive2
    #
    naive2_mase = mase(naive2_prediction, test_set.data, training_set.masep)
    naive2_smape = smape(naive2_prediction, test_set.data)

    model_mase = mase(model_prediction, test_set.data, training_set.masep)
    model_smape = smape(model_prediction, test_set.data)

    # summary
    smape_summary = weighted_average(model_smape, training_set.info, index_name='sMAPE')
    owa_summary = weighted_average(owa(model_mase, model_smape, naive2_mase, naive2_smape), index_name='OWA')

    return pd.concat([smape_summary, owa_summary])


def mase(prediction: np.ndarray, target: np.ndarray, masep: np.ndarray) -> np.ndarray:
    return np.array([np.mean(ts) for ts in np.abs(prediction - target) / masep])


def smape(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([np.mean(ts) for ts in (200 * np.abs(prediction - target) / (np.abs(target) + np.abs(prediction)))])


def owa(model_mase: np.ndarray, model_smape: np.ndarray, naive2_mase: np.ndarray, naive2_smape: np.ndarray):
    return (model_mase / naive2_mase + model_smape / naive2_smape) / 2


def weighted_average(scores: np.ndarray, m4_info: M4Info, index_name: str) -> pd.DataFrame:
    assert (len(scores) == len(m4_info.ids))
    seasonal_patterns = m4_info.data.SP.drop_duplicates().values
    grouped_scores = OrderedDict(list(zip(seasonal_patterns, [[]] * len(seasonal_patterns))))

    for i, sp in enumerate(m4_info.data.SP):
        grouped_scores[sp].append(scores[i])

    weighted_avg_scores = {'Others': 0.0}
    len_others = len(
        m4_info.data[(m4_info.data.SP == 'Weekly') | (m4_info.data.SP == 'Daily') | (m4_info.data.SP == 'Hourly')])
    for sp, values in grouped_scores.items():
        if sp == 'Yearly' or sp == 'Quarterly' or sp == 'Monthly':
            weighted_avg_scores[sp] = (np.array(values).mean() * len(values)) / len(m4_info.ids)
        else:
            weighted_avg_scores['Others'] += (np.array(values).mean() * (len(values) / len_others)) / len(m4_info.ids)

    return pd.DataFrame(weighted_avg_scores, index=[index_name])

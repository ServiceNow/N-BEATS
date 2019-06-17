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

    model_mase_summary = weighted_average(scores=mase(model_prediction, test_set.data, training_set.masep),
                                          m4_info=training_set.info,
                                          index_name='MASE')
    naive2_mase_summary = weighted_average(scores=mase(naive2_prediction, test_set.data, training_set.masep),
                                           m4_info=training_set.info,
                                           index_name='MASE_naive2')
    model_smape_summary = weighted_average(scores=smape(model_prediction, test_set.data),
                                           m4_info=training_set.info,
                                           index_name='sMAPE')
    naive2_smape_summary = weighted_average(scores=smape(naive2_prediction, test_set.data),
                                            m4_info=training_set.info,
                                            index_name='sMAPE_naive2')

    owa_score = owa(model_mase_summary.values,
                    model_smape_summary.values,
                    naive2_mase_summary.values,
                    naive2_smape_summary.values)
    owa_summary = pd.DataFrame(owa_score, columns=model_smape_summary.columns)
    owa_summary.index = ['OWA']

    return pd.concat([model_smape_summary, owa_summary])


def mase(prediction: np.ndarray, target: np.ndarray, masep: np.ndarray) -> np.ndarray:
    return np.array([np.mean(ts) for ts in (np.abs(prediction - target) / masep)])


def smape(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([np.mean(ts) for ts in (200 * np.abs(prediction - target) / (np.abs(target) + np.abs(prediction)))])


def owa(model_mase: np.ndarray, model_smape: np.ndarray, naive2_mase: np.ndarray, naive2_smape: np.ndarray):
    return (model_mase / naive2_mase + model_smape / naive2_smape) / 2


def weighted_average(scores: np.ndarray, m4_info: M4Info, index_name: str) -> pd.DataFrame:
    assert (len(scores) == len(m4_info.ids))
    seasonal_patterns = m4_info.data.SP.drop_duplicates().values
    grouped_scores = OrderedDict(list(zip(seasonal_patterns, [np.array([])] * len(seasonal_patterns))))

    for i, sp in enumerate(m4_info.data.SP):
        grouped_scores[sp] = np.append(grouped_scores[sp], scores[i])

    weighted_avg_scores = OrderedDict([('Yearly', 0.0), ('Quarterly', 0.0), ('Monthly', 0.0), ('Others', 0.0),
                                       ('Average', 0.0)])
    len_others = len(
        m4_info.data[(m4_info.data.SP == 'Weekly') | (m4_info.data.SP == 'Daily') | (m4_info.data.SP == 'Hourly')])
    for sp, values in grouped_scores.items():
        if sp == 'Yearly' or sp == 'Quarterly' or sp == 'Monthly':
            weighted_avg_scores[sp] = np.array(values).mean()
        else:
            weighted_avg_scores['Others'] += np.array(values).mean() * (len(values) / len_others)

    weighted_avg_scores['Average'] = (weighted_avg_scores['Yearly'] * len(m4_info.data[m4_info.data.SP == 'Yearly']) +
                                      weighted_avg_scores['Quarterly'] * len(
                m4_info.data[m4_info.data.SP == 'Quarterly']) +
                                      weighted_avg_scores['Monthly'] * len(m4_info.data[m4_info.data.SP == 'Monthly']) +
                                      weighted_avg_scores['Others'] * len_others) / len(m4_info.ids)

    return pd.DataFrame(weighted_avg_scores, index=[index_name])

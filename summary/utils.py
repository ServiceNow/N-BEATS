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
Frequently used functions for building summaries.
"""
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

def median_ensemble(experiment_path: str,
                    summary_filter: str = '**',
                    forecast_file: str = 'forecast.csv',
                    group_by: str = 'id'):
    """
    Build a median ensemble from files found in the experiment path.

    :param experiment_path: Experiment path.
    :param summary_filter: Filter which experiment instances should be included in ensemble.
    :param forecast_file: Name of the file with results.
    :param group_by: Grouping key.
    :return: Pandas dataframe with median forecasts.
    """
    return pd.concat([pd.read_csv(file)
                      for file in
                      tqdm(glob(os.path.join(experiment_path, summary_filter, forecast_file)))], sort=False) \
        .set_index(group_by).groupby(level=group_by, sort=False).median().values

def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.

    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])

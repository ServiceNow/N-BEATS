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
M3 Summary
"""
from collections import OrderedDict
from typing import Dict

import numpy as np

from common.metrics import smape_1
from datasets.m3 import M3Dataset, M3Meta
from summary.utils import group_values

class M3Summary:
    def __init__(self):
        self.test_set = M3Dataset.load(training=False)

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecasts using M3 test dataset.

        :param forecast: Forecasts. Shape: timeseries, horizon.
        :return: sMAPE grouped by seasonal patterns.
        """
        results = OrderedDict()
        cumulative_metrics = 0
        cumulative_points = 0
        offset = 0
        for sp in M3Meta.seasonal_patterns:
            target = group_values(self.test_set.values, self.test_set.groups, sp)
            sp_forecast = group_values(forecast, self.test_set.groups, sp)
            sp_smape = smape_1(sp_forecast, target)
            cumulative_metrics += np.sum(sp_smape)
            cumulative_points += np.prod(target.shape)
            results[sp] = round(float(np.mean(sp_smape)), 2)
            offset += len(target)

        results['Average'] = round(cumulative_metrics / cumulative_points, 2)
        return results

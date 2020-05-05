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

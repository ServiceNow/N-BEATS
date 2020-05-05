"""
Tourism summary
"""
from collections import OrderedDict
from typing import Dict

import numpy as np

from common.metrics import mape
from datasets.tourism import TourismDataset, TourismMeta
from summary.utils import group_values

class TourismSummary:

    def __init__(self):
        self.test_set = TourismDataset.load(training=False)

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecasts for Tourism dataset.

        :param forecast: Forecasts. Shape: timeseries, time
        :return: MAPE for each seasonal pattern.
        """
        results = OrderedDict()
        cumulative_metrics = 0
        cumulative_points = 0
        offset = 0
        for sp in TourismMeta.seasonal_patterns:
            target = group_values(self.test_set.values, self.test_set.groups, sp)
            sp_forecast = group_values(forecast, self.test_set.groups, sp)
            score = mape(sp_forecast, target)
            cumulative_metrics += np.sum(score)
            cumulative_points += np.prod(target.shape)
            results[sp] = round(float(np.mean(score)), 2)
            offset += len(target)

        results['Average'] = round(cumulative_metrics / cumulative_points, 2)
        return results
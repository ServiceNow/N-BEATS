"""
Electricity summary.
"""
from typing import Callable, Dict

import numpy as np

from datasets.electricity import ElectricityDataset

class ElectricitySummary:
    def __init__(self, test_set: ElectricityDataset):
        self.target_values = test_set.values

    def evaluate(self, forecast: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, float]:
        """
        Evaluate forecasts for Electricity dataset using provided metric.

        :param forecast: Forecasts. Shape: timeseries, time.
        :param metric: Metric function which takes forecast and target values and returns the overall score.
        :return: Dictionary which contains the score rounded to 3 decimal points.
        """
        return {'all': np.round(metric(forecast, self.target_values), 3)}
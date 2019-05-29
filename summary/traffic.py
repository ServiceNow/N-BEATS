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
Traffic summary.
"""
from typing import Callable, Dict

import numpy as np

from datasets.traffic import TrafficDataset

class TrafficSummary:
    def __init__(self, test_set: TrafficDataset):
        self.target_values = test_set.values

    def evaluate(self, forecast: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, float]:
        """
        Evaluate forecasts for Traffic dataset using provided metric.

        :param forecast: Forecasts. Shape: timeseries, time.
        :param metric: Metric function which takes forecast and target values and returns the overall score.
        :return: Dictionary which contains the score rounded to 3 decimal points.
        """
        return {'all': np.round(metric(forecast, self.target_values), 3)}
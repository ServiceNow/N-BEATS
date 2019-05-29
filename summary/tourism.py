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
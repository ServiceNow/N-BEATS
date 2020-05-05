"""
Tourism summary unit test.
"""
import unittest

import numpy as np

from datasets.tourism import TourismDataset, TourismMeta
from summary.tourism import TourismSummary
from summary.utils import group_values


class TestTourismSummary(unittest.TestCase):

    def test_evaluation(self):
        train_dataset = TourismDataset.load(training=True)

        naive_forecasts = []
        for seasonal_pattern in TourismMeta.seasonal_patterns:
            train_values = group_values(train_dataset.values, train_dataset.groups, seasonal_pattern)
            for ts in train_values:
                naive_forecast = self.snaive(ts,
                                             horizon=TourismMeta.horizons_map[seasonal_pattern],
                                             seasonality=TourismMeta.frequency_map[seasonal_pattern])
                naive_forecasts.append(naive_forecast)
        summary = TourismSummary()
        naive_summary = summary.evaluate(np.array(naive_forecasts))

        # From https://robjhyndman.com/papers/forecompijf.pdf
        self.assertEqual(naive_summary["Yearly"], 23.61)
        self.assertEqual(naive_summary["Monthly"], 22.56)
        self.assertEqual(naive_summary["Quarterly"], 16.46)

    def snaive(self, insample, horizon, seasonality=1):
        forecast = np.zeros((horizon,))
        for i in range(horizon):
            idx = len(insample) + i % seasonality - seasonality
            forecast[i] = insample[idx]
        return forecast


if __name__ == '__main__':
    unittest.main()

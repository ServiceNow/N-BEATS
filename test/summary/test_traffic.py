"""
Traffic summary unit tests.
"""
import unittest
import numpy as np

from datasets.traffic import TrafficDataset, TrafficMeta
from summary.traffic import TrafficSummary
from common.metrics import nd, nrmse


class TestTrafficSummary(unittest.TestCase):

    def test_naive(self):
        dataset = TrafficDataset.load()
        test_windows = 7
        training_set, test_set = dataset.split_by_date('2009-03-24 00')
        training_values = training_set.values
        test_values = test_set.values

        summary = TrafficSummary(test_set=test_set)

        mean_forecasts = []
        for i in range(test_windows):
            window_training_set = np.concatenate(
                [training_values, test_values[:, :i * TrafficMeta.horizon]], axis=1)
            window_forecast = np.repeat(np.repeat(np.mean(window_training_set), TrafficMeta.lanes, axis=0)[:, None],
                                        TrafficMeta.horizon, axis=1)
            mean_forecasts = window_forecast if len(mean_forecasts) == 0 else np.concatenate(
                [mean_forecasts, window_forecast], axis=1)
        nd_value = summary.evaluate(mean_forecasts, nd)['metric']
        nrmse_value = summary.evaluate(mean_forecasts, nrmse)['metric']

        # as per Table 2 in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
        self.assertTrue(np.allclose(nd_value, 0.56))
        self.assertTrue(np.allclose(nrmse_value, 0.826))


if __name__ == '__main__':
    unittest.main()
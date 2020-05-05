"""
Electricity summary unit tests.
"""
import unittest
import numpy as np

from datasets.electricity import ElectricityDataset, ElectricityMeta
from summary.electricity import ElectricitySummary
from common.metrics import nd, nrmse


class TestElectricitySummary(unittest.TestCase):

    def test_naive(self):
        dataset = ElectricityDataset.load()
        test_windows = 7
        training_set, test_set = dataset.split_by_date('2014-12-25 00')
        training_values = training_set.values
        test_values = test_set.values

        summary = ElectricitySummary(test_set=test_set)

        mean_forecasts = []
        for i in range(test_windows):
            window_training_set = np.concatenate(
                [training_values, test_values[:, :i * ElectricityMeta.horizon]], axis=1)
            window_forecast = np.repeat(np.repeat(np.mean(window_training_set), ElectricityMeta.clients, axis=0)[:, None],
                                        ElectricityMeta.horizon, axis=1)
            mean_forecasts = window_forecast if len(mean_forecasts) == 0 else np.concatenate(
                [mean_forecasts, window_forecast], axis=1)
        nd_value = summary.evaluate(mean_forecasts, nd)['metric']
        nrmse_value = summary.evaluate(mean_forecasts, nrmse)['metric']

        # as per Table 2 in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
        self.assertTrue(np.allclose(nd_value, 1.410))
        self.assertTrue(np.allclose(nrmse_value, 4.528))


if __name__ == '__main__':
    unittest.main()
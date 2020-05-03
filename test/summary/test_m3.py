"""
M3 Summary unit test
"""
import os
import unittest

import numpy as np
import pandas as pd

from common.http_utils import download, url_file_name
from common.settings import TESTS_STORAGE
from summary.m3 import M3Summary

FORECASTS_URL = 'https://forecasters.org/data/m3comp/M3Forecast.xls'
FORECASTS_FILE_PATH = os.path.join(TESTS_STORAGE, 'm3', url_file_name(FORECASTS_URL))

class TestM3Summary(unittest.TestCase):
    def setUp(self) -> None:
        download(FORECASTS_URL, FORECASTS_FILE_PATH)

    def test_summary(self):
        summary = M3Summary()
        naive2 = pd.read_excel(FORECASTS_FILE_PATH, sheet_name='NAIVE2', header=None)
        naive2_forecast = np.array([ts[~np.isnan(ts)]
                                    for ts in naive2[naive2.columns[2:]].values])
        result = summary.evaluate(naive2_forecast)

        # based on http://www.forecastingprinciples.com/paperpdf/Makridakia-The%20M3%20Competition.pdf
        # Tables 13-16 and Table 6 for Average.
        self.assertEqual(result['M3Year'], 17.88)
        self.assertEqual(result['M3Quart'], 9.95)
        self.assertEqual(result['M3Month'], 16.91)
        self.assertEqual(result['M3Other'], 6.3)
        self.assertEqual(result['Average'], 15.47)

if __name__ == '__main__':
    unittest.main()
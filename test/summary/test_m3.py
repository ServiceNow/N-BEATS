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
M3 summary unit test
"""
import os
import unittest

import numpy as np
import pandas as pd

from common.http_utils import download, url_file_name
from common.settings import TESTS_STORAGE_PATH
from summary.m3 import M3Summary

FORECASTS_URL = 'https://forecasters.org/data/m3comp/M3Forecast.xls'
FORECASTS_FILE_PATH = os.path.join(TESTS_STORAGE_PATH, 'm3', url_file_name(FORECASTS_URL))

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
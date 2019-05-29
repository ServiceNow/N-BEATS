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
M4 summary unit test.
"""
import os
import unittest

import pandas as pd
import patoolib

from common.http_utils import download, url_file_name
from common.settings import TESTS_STORAGE_PATH
from summary.m4 import M4Summary

WINNER_FORECAST_URL = 'https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-118.rar'
TEST_STORAGE_PATH = os.path.join(TESTS_STORAGE_PATH, 'm4')
WINNER_FORECAST_PATH = os.path.join(TEST_STORAGE_PATH, 'submission-118.csv')

class TestM4Summary(unittest.TestCase):
    def setUp(self) -> None:
        winner_archive = os.path.join(TEST_STORAGE_PATH, url_file_name(WINNER_FORECAST_URL))
        download(WINNER_FORECAST_URL, winner_archive)
        if not os.path.isfile(WINNER_FORECAST_PATH):
            patoolib.extract_archive(winner_archive, outdir=TEST_STORAGE_PATH)

    def test_evaluation(self):
        # https://www.researchgate.net/profile/Spyros_Makridakis/publication/325901666_The_M4_Competition_Results_\
        # findings_conclusion_and_way_forward/links/5b2c9aa4aca2720785d66b5e/The-M4-Competition-Results-findings-\
        # conclusion-and-way-forward.pdf?origin=publication_detail

        m4_summary = M4Summary()
        m4_winner_forecast = pd.read_csv(WINNER_FORECAST_PATH)
        m4_winner_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
        smape_results, owa_results = m4_summary.evaluate(m4_winner_forecast.values)

        # Results are based on Table 1 of:
        # https://www.researchgate.net/profile/Spyros_Makridakis/publication/325901666_The_M4_Competition_Results_
        # findings_conclusion_and_way_forward/links/5b2c9aa4aca2720785d66b5e/The-M4-Competition-Results-findings-
        # conclusion-and-way-forward.pdf?origin=publication_detail
        self.assertEqual(smape_results['Yearly'], 13.176)
        self.assertEqual(smape_results['Quarterly'], 9.679)
        self.assertEqual(smape_results['Monthly'], 12.126)
        self.assertEqual(smape_results['Others'], 4.013)
        self.assertEqual(smape_results['Average'], 11.374)

        self.assertEqual(owa_results['Yearly'], 0.778)
        self.assertEqual(owa_results['Quarterly'], 0.847)
        self.assertEqual(owa_results['Monthly'], 0.836)
        self.assertEqual(owa_results['Others'], 0.920)
        self.assertEqual(owa_results['Average'], 0.821)


if __name__ == '__main__':
    unittest.main()
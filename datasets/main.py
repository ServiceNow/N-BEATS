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
Datasets module
"""
import logging
import ssl

from fire import Fire

from datasets.electricity import ElectricityDataset
from datasets.m3 import M3Dataset
from datasets.m4 import M4Dataset
from datasets.tourism import TourismDataset
from datasets.traffic import TrafficDataset

def build():
    """
    Download all datasets.
    """

    logging.info('M4 Dataset')
    M4Dataset.download()

    logging.info('\n\nM3 Dataset')
    M3Dataset.download()

    logging.info('\n\nTourism Dataset')
    TourismDataset.download()

    logging.info('\n\nElectricity Dataset')
    ElectricityDataset.download()

    logging.info('\n\nTraffic Dataset')
    TrafficDataset.download()

if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()
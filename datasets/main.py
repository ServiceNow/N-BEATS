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
    # Fix for: Hostname mismatch, certificate is not valid for 'mcompetitions.unic.ac.cy'
    ssl._create_default_https_context = ssl._create_unverified_context

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
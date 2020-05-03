"""
Common settings
"""
import os

STORAGE=os.getenv('STORAGE')
DATASETS_DIR=os.path.join(STORAGE, 'datasets')
EXPERIMENTS_DIR=os.path.join(STORAGE, 'experiments')
TESTS_STORAGE=os.path.join(STORAGE, 'test')
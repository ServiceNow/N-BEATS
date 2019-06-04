import os

#
# Dataset URLs
#
M4_TRAINING_SET_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2017/12/M4DataSet.zip'
M4_TEST_SET_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2018/07/M-test-set.zip'
M4_INFO_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2018/12/M4Info.csv'


#
# Settings for default environment (see Docker setup in README)
#
DEFAULT_DATA_DIR = os.path.join(os.sep, 'project', 'dataset')
DEFAULT_EXPERIMENTS_DIR = os.path.join(os.sep, 'project', 'experiments')

#
# Training settings
#
INPUT_MAXSIZE = 1000

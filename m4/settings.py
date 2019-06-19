import os

#
# Dataset URLs
#
M4_TRAINING_SET_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2017/12/M4DataSet.zip'
M4_TEST_SET_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2018/07/M-test-set.zip'
M4_INFO_URL = 'https://www.m4.unic.ac.cy/wp-content/uploads/2018/12/M4Info.csv'
M4_NAIVE2_URL = 'https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-Naive2.rar'


#
# Settings for default environment (see Docker setup in README)
#
M4_DATA_DIR = os.path.join(os.sep, 'project', 'dataset', 'm4')
M4_EXPERIMENTS_DIR = os.path.join(os.sep, 'project', 'experiments', 'm4')
M4_PREDICTION_FILE_NAME = 'predictions.csv'

#
# Training settings
#
M4_INPUT_MAXSIZE = 1000
# sampling window dictionary is: M4 Horizon -> how many horizons of past can be used for sampling input windows
M4_SAMPLING_WINDOW_LIMIT = {6: 1.5, 8: 1.5, 18: 1.5, 13: 15.0, 14: 15.0, 48: 15.0}

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


'''
notations: 
    in H_0_1, zero is upperscripts and one is subscripts
'''

ROOT_DIR = os.path.dirname(__file__)

### data 0.5_1
# DATA_PATH = ROOT_DIR + '/marker_data/Trimmed_lzj_2021_12_02_tredmill_0.5_1.trc'
# SAVE_DIR = ROOT_DIR + '/processed_0.5_1'

### data 0.8_11
# DATA_PATH = ROOT_DIR + '/marker_data/Trimmed_lzj_2021_12_02_tredmill_0.8_11_8500_16300.trc'
# SAVE_DIR = ROOT_DIR + '/processed_0.8_11'

### 1.1_11_3300_12100
# DATA_PATH = ROOT_DIR + '/marker_data/Trimmed_lzj_2021_12_02_tredmill_1.1_11_3300_12100.trc'
# SAVE_DIR = ROOT_DIR + '/processed_1.1_11_3300_12100'


# ## data 0.8_21
DATA_PATH = ROOT_DIR + '/marker_data/Trimmed_lzj_2021_12_02_tredmill_0.8_21_1500_9300.trc'
SAVE_DIR = ROOT_DIR + '/processed_0.8_21'




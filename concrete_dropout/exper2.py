# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from evaluation2 import *

#==============================================================================
# BlogFeedback (blog)

data = pd.read_csv('../data/blog.csv')
data = data.dropna()

var_cont = data.columns[0:276]
var_cate = []
y = data.columns[-1]

aver_calibrate("blog", data, var_cont, var_cate, y)

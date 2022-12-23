# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from evaluation_fig import *

#==============================================================================
# Electrical Grid Stability

data = pd.read_csv('../../data/Data_for_UCI_named.csv')
data = data.drop(['stabf'], axis = 1)
data = data.dropna()

var_cont = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']
var_cate = []
y = np.array(['stab'])[0]

plot_split("Electrical Grid Stability", data, var_cont, var_cate, y)

#==============================================================================
# Appliances energy prediction

data = pd.read_csv('../../data/energydata_complete.csv')
data = data.drop(['date'], axis = 1)
data = data.dropna()

var_cont = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 
                      'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 
                      'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 
                      'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']
var_cate = []
y = 'Appliances'

plot_split("Appliances energy prediction", data, var_cont, var_cate, y)

#==============================================================================
# Physicochemical Properties

data = pd.read_csv('../../data/CASP.csv')
data = data.dropna()

var_cont = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
var_cate = []
y = np.array(['RMSD'])[0]

plot_split("Physicochemical Properties", data, var_cont, var_cate, y)

#==============================================================================
# Facebook Comment Volume (facebook1)

data = pd.read_csv('../../data/facebook1.csv')
data = data.dropna()

var_cont = data.columns[0:52]
var_cate = []
y = data.columns[-1]

plot_split("facebook1", data, var_cont, var_cate, y)

#==============================================================================
# Wave Energy Converters (wave)

data = pd.read_csv('../../data/wave.csv')
data = data.dropna()

var_cont = data.columns[0:48]
var_cate = []
y = data.columns[-1]

plot_split("wave", data, var_cont, var_cate, y)

#==============================================================================
# YearPredictionMSD (year)

data = pd.read_csv('../../data/year.csv')
data = data.dropna()
data = data.sample(n = 100000, random_state = 42) 

var_cont = data.columns[0:90]
var_cate = []
y = np.array(['0'])[0]

plot_split("year", data, var_cont, var_cate, y)

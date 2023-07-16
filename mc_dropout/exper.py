# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from evaluation import *

#==============================================================================
# Electrical Grid Stability

data = pd.read_csv('../data/Data_for_UCI_named.csv')
data = data.drop(['stabf'], axis = 1)
data = data.dropna()

var_cont = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']
var_cate = []
y = np.array(['stab'])[0]

aver_calibrate("Electrical Grid Stability", data, var_cont, var_cate, y)

#==============================================================================
# Conditional Based Maintenance

data = pd.read_table('../data/CBM_data.txt', sep = '   ')
data.columns = ['lp', 'v', 'GTT', 'GTn', 'GGn','Ts','Tp','T48','T1','T2','P48','P1','P2','Pexh','TIC','mf','GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']
data = data.drop(['GT Turbine decay state coefficient', 'T1'], axis = 1)
data = data.dropna()

var_cont = ['lp', 'v', 'GTT', 'GTn', 'GGn','Ts','Tp','T48','T2','P48','P1','P2','Pexh','TIC','mf']
var_cate = []
y = np.array(['GT Compressor decay state coefficient'])[0]

aver_calibrate("Conditional Based Maintenance", data, var_cont, var_cate, y)

#==============================================================================
# Appliances energy prediction

data = pd.read_csv('../data/energydata_complete.csv')
data = data.drop(['date'], axis = 1)
data = data.dropna()

var_cont = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 
                      'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 
                      'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 
                      'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']
var_cate = []
y = 'Appliances'

aver_calibrate("Appliances energy prediction", data, var_cont, var_cate, y)

#==============================================================================
# Real-time Election

data = pd.read_csv('../data/ElectionData.csv')
data = data.drop(['TimeElapsed', 'time', 'territoryName', 'Party'], axis = 1)
data = data.dropna()

var_cont = ['totalMandates', 'availableMandates', 'numParishes', 'numParishesApproved', 'blankVotes', 'blankVotesPercentage', 'nullVotes', 'nullVotesPercentage', 'votersPercentage', 'subscribedVoters', 'totalVoters', 'pre.blankVotes', 'pre.blankVotesPercentage', 'pre.nullVotes', 'pre.nullVotesPercentage', 'pre.votersPercentage', 'pre.subscribedVoters', 'pre.totalVoters', 'Mandates', 'Percentage', 'validVotesPercentage', 'Votes', 'Hondt']
var_cate = []
y = np.array(['FinalMandates'])[0]

aver_calibrate("Real-time Election", data, var_cont, var_cate, y)

#==============================================================================
# Industry Energy Consumption

data = pd.read_csv('../data/Steel_industry_data.csv')
data = data.drop(['date'], axis = 1)
data = data.dropna()

var_cont = ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']
var_cate = ['WeekStatus', 'Day_of_week', 'Load_Type']
y =  np.array(['Usage_kWh'])[0]

aver_calibrate("Industry Energy Consumption", data, var_cont, var_cate, y)

#==============================================================================
# Beijing PM2.5

data = pd.read_csv('../data/PRSA_data_2010.1.1-2014.12.31.csv')
data = data.drop(['No'], axis = 1)
data = data.dropna()

var_cont = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
var_cate = ['year', 'cbwd']
y = 'pm2.5'

aver_calibrate("Beijing PM2.5", data, var_cont, var_cate, y)

#==============================================================================
# Physicochemical Properties

data = pd.read_csv('../data/CASP.csv')
data = data.dropna()

var_cont = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
var_cate = []
y = np.array(['RMSD'])[0]

aver_calibrate("Physicochemical Properties", data, var_cont, var_cate, y)

#==============================================================================
# Traffic Volume

data = pd.read_csv('../data/Metro_Interstate_Traffic_Volume.csv')
data = data.drop(['weather_description', 'date_time'], axis = 1)
data = data.dropna()

var_cont = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
var_cate = ['holiday', 'weather_main']
y = np.array(['traffic_volume'])[0]

aver_calibrate("Traffic Volume", data, var_cont, var_cate, y)

#==============================================================================
# Power consumption of T

data = pd.read_csv('../data/Tetuan City power consumption.csv')
data = data.drop(['DateTime', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption'], axis = 1)
data = data.dropna()

var_cont = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
var_cate = []
y = 'Zone 1 Power Consumption'

aver_calibrate("Power consumption of T", data, var_cont, var_cate, y)

#==============================================================================
# Online Video

data = pd.read_csv('../data/transcoding_mesurment.tsv', sep = '\t')
data = data.drop(['id', 'umem'], axis = 1)
data = data.dropna()

var_cont = ['duration', 'width', 'height', 'bitrate', 'framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size', 'size', 'o_bitrate', 'o_framerate', 'o_width', 'o_height']
var_cate = ['codec', 'o_codec']
y = np.array(['utime'])[0]

aver_calibrate("Online Video", data, var_cont, var_cate, y)

#==============================================================================
# GPU kernel performance

data = pd.read_csv('../data/sgemm_product.csv')
data = data.drop(['Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis = 1)
data = data.dropna()

var_cont = ['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB', 'KWI', 'VWM', 'VWN']
var_cate = ['STRM', 'STRN', 'SA', 'SB']
y = np.array(['Run1 (ms)'])[0]

aver_calibrate("GPU kernel performance", data, var_cont, var_cate, y)

#==============================================================================
# Query Analytics

data = pd.read_csv('../data/Range-Queries-Aggregates.csv')
data = data.drop(['Unnamed: 0', 'count', 'sum_'], axis = 1)
data = data.dropna()

var_cont = ['x', 'y', 'x_range', 'y_range']
var_cate = []
y = np.array(['avg'])[0]

aver_calibrate("Query Analytics", data, var_cont, var_cate, y)

#==============================================================================
# Facebook Comment Volume (facebook1)

data = pd.read_csv('../data/facebook1.csv')
data = data.dropna()

var_cont = data.columns[0:52]
var_cate = []
y = data.columns[-1]

aver_calibrate("facebook1", data, var_cont, var_cate, y)

#==============================================================================
# Facebook Comment Volume (facebook2)

data = pd.read_csv('../data/facebook2.csv')
data = data.dropna()

var_cont = data.columns[0:52]
var_cate = []
y = data.columns[-1]

aver_calibrate("facebook2", data, var_cont, var_cate, y)

#==============================================================================
# BlogFeedback (blog)

data = pd.read_csv('../data/blog.csv')
data = data.dropna()

var_cont = data.columns[0:276]
var_cate = []
y = data.columns[-1]

aver_calibrate("blog", data, var_cont, var_cate, y)

#==============================================================================
# Wave Energy Converters (wave)

data = pd.read_csv('../data/wave.csv')
data = data.dropna()

var_cont = data.columns[0:48]
var_cate = []
y = data.columns[-1]

aver_calibrate("wave", data, var_cont, var_cate, y)

#==============================================================================
# YearPredictionMSD (year)

data = pd.read_csv('../data/year.csv')
data = data.dropna()
data = data.sample(n = 100000, random_state = 42) 

var_cont = data.columns[0:90]
var_cate = []
y = np.array(['0'])[0]

aver_calibrate("year", data, var_cont, var_cate, y)

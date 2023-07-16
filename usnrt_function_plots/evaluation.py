# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import usnrt

def rmse(x, y):
    x_ = np.array(x).reshape(-1)
    y_ = np.array(y).reshape(-1)
    return np.sqrt(((x_ - y_) ** 2).mean())

#==============================================================================

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_ECE(true, pred, tau):
    ece = np.mean(np.abs(np.mean(np.array(true).reshape(-1, 1) < pred, axis = 0) - tau.reshape(-1)))
    return ece

#==============================================================================

def calibration(data_name, train_data, test_data, var_cont, var_cate, y, set_seed):
    if not os.path.isdir("results/"+data_name):
        os.mkdir("results/"+data_name)
    data_dir = "results/"+data_name + '/'
    
    tau = np.array(range(1, 100)) / 100
    
#==============================================================================
    
    method_name = 'usnrt'
    
    split_num = 10
    num_epochs = 1000
    batch_size = 64
    lr = 0.01
    train_prop = 0.8
    
    in_dim = len(var_cont) + len(var_cate)
    dims = [[8, 4], [4, 2]]
    split_dims = list(np.array(dims[0]) * in_dim)
    leaf_dims = list(np.array(dims[1]) * in_dim)
    
    p_value = 0.01
    Nsplit = 10
    Nmin = max(math.floor(len(train_data) / Nsplit), 1000)
    
    our_tree = usnrt.grow_tree(train_data, var_cont, var_cate, y, split_dims, leaf_dims, p_value, Nmin,
                        split_num, num_epochs, batch_size, lr, train_prop, set_seed = set_seed,
                        index = [1], region_info = 'All Data')
    mu_ours = our_tree.predict_all(test_data.drop(y, axis = 1), var_cont, var_cate)
    var_ours = our_tree.uncertainty_all(test_data.drop(y, axis = 1), var_cont, var_cate)
    quantiles = gen_quantiles(mu_ours, var_ours, tau)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,split_dims,leaf_dims,p_value,Nsplit,ECE,rmse,sigma2,num_points\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},\"{}\",\"{}\",{},{},{},{},\"{}\",\"{}\"\n".format(set_seed, str(dims[0]), str(dims[1]), p_value, Nsplit,
            cal_ECE(test_data[y], quantiles, tau), rmse(mu_ours, test_data[y]), our_tree.getSigma2(), our_tree.getNumPoints()))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame([mu_ours, var_ours, test_data[y].tolist()]).T
    results.columns = ['mean', 'variance', 'true']
    results.to_csv(data_dir + method_name + "/" + "{}_{}_{}_{}_{}.csv".format(set_seed, str(dims[0]), str(dims[1]), p_value, Nsplit), index = False)
    
    if not os.path.isdir(data_dir + method_name + "_plot/"):
        os.mkdir(data_dir + method_name + "_plot/")
    plotDir = data_dir + method_name + "_plot/" + "{}_{}_{}_{}_{}/".format(set_seed, str(dims[0]), str(dims[1]), p_value, Nsplit)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    artData = pd.DataFrame(train_data[list(var_cont)+list(var_cate)], copy = True)
    for variable in var_cont:
        artData[variable] = train_data[variable].mean()
    for variable in var_cate:
        artData[variable] = train_data[variable].value_counts().idxmax()
    artData = artData.iloc[:1000]
    
    for variable in var_cont:
        meanValue = artData[variable].iloc[0]
        maxValue = np.percentile(train_data[variable], 99)
        minValue = np.percentile(train_data[variable], 1)
        artData[variable] = np.linspace(minValue, maxValue, artData.shape[0])
        
        pdf = PdfPages(plotDir+variable+'.pdf')
        plt.figure(figsize=(7,4),dpi=300)
        
        sigmas = np.sqrt(our_tree.uncertainty_all(artData,var_cont,var_cate))
        plt.scatter(artData[variable], sigmas, c='black', s=0.1)
        plt.xlabel('Variable: '+variable)
        plt.ylabel('Sigma')
        plt.title("Dataset: "+data_name)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        pdf.close()
        artData[variable] = meanValue

#==============================================================================

def aver_calibrate(data_name, data, var_cont, var_cate, y):
    if not os.path.isdir("results/"):
        os.mkdir("results/")
    print('')
    print(data_name)
    print(len(data), len(var_cont) + len(var_cate))
    
    N_aver = 1#5 if len(data) < 1e5 else 1
    Train_proportion = 0.8 if len(data) < 3e5 else 0.5
    
    data[var_cont] = (data[var_cont] - data[var_cont].mean()) / data[var_cont].std()
    data[y] = (data[y] - data[y].mean()) / data[y].std()
    
    for i in range(N_aver):
        set_seed = 42 + 100 * i
        
        random.seed(set_seed)
        train_index = random.sample(range(len(data)), math.floor(len(data) * Train_proportion))
        test_index = list(set(range(len(data))) - set(train_index))
        
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        calibration(data_name, train_data, test_data, var_cont, var_cate, y, set_seed)
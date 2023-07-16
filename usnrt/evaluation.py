# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

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
    
#==============================================================================
# Our Tree Ensemble
    
    method_name = 'usnrt_ensemble'

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
    
    tree_Num = 5
    mu_ensem = np.zeros((len(test_data), tree_Num))
    var_ensem = np.zeros((len(test_data), tree_Num))
    mu_ensem[:, 0] = mu_ours
    var_ensem[:, 0] = var_ours
    
    for n in range(1, tree_Num):
        our_tree = usnrt.grow_tree(train_data, var_cont, var_cate, y, split_dims, leaf_dims, p_value, Nmin,
                            split_num, num_epochs, batch_size, lr, train_prop, set_seed = set_seed + n * 1997,
                            index = [1], region_info = 'All Data')
        mu_ensem[:, n] = our_tree.predict_all(test_data.drop(y, axis = 1), var_cont, var_cate)
        var_ensem[:, n] = our_tree.uncertainty_all(test_data.drop(y, axis = 1), var_cont, var_cate)
    
    results = pd.DataFrame(mu_ensem, copy = True)
    results.columns = ['mean'+str(x) for x in range(tree_Num)]
    results[['variance'+str(x) for x in range(tree_Num)]] = var_ensem
    
    var_ensem2 = var_ensem.mean(axis = 1)
    var_ensem = (var_ensem + mu_ensem ** 2).mean(axis = 1) - mu_ensem.mean(axis = 1) ** 2
    mu_ensem = mu_ensem.mean(axis = 1)
    quantiles = gen_quantiles(mu_ensem, var_ensem, tau)
    quantiles2 = gen_quantiles(mu_ensem, var_ensem2, tau)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,split_dims,leaf_dims,p_value,Nsplit,tree_Num,ECE,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},\"{}\",\"{}\",{},{},{},{},{}\n".format(set_seed, str(dims[0]), str(dims[1]), p_value,
            Nsplit, tree_Num, cal_ECE(test_data[y], quantiles, tau), rmse(mu_ensem, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results['mean'] = mu_ensem
    results['variance'] = var_ensem
    results['variance_a'] = var_ensem2
    results['true'] = test_data[y].tolist()
    results.to_csv(data_dir + method_name + "/" + "{}_{}_{}_{}_{}_{}.csv".format(set_seed,
                    str(dims[0]), str(dims[1]), p_value, Nsplit, tree_Num), index = False)

#==============================================================================

def aver_calibrate(data_name, data, var_cont, var_cate, y):
    if not os.path.isdir("results/"):
        os.mkdir("results/")
    print('')
    print(data_name)
    print(len(data), len(var_cont) + len(var_cate))
    
    N_aver = 5 if len(data) < 1e5 else 1
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
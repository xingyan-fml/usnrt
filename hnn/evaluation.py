# -*- coding: utf-8 -*-

import os
import math
import torch
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

from fit_hnn import *

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
# one-hot encoding
    
    value_sets = {}
    dummy_rules = {}
    for variable in var_cate:
        value_set = sorted(list(set(train_data[variable])))
        if len(value_set) <= 1 or len(value_set) > 10:
            continue
        value_sets[variable] = value_set
        dummies = np.identity(len(value_set))[:, 1:].tolist()
        dummy_rules[variable] = dict(zip(value_set, dummies))
    
    train_data_onehot = pd.DataFrame(train_data[list(var_cont)], copy = True)
    test_data_onehot = pd.DataFrame(test_data[list(var_cont)], copy = True)
    
    for variable in var_cate:
        if variable in dummy_rules:
            new_columns = [str(variable) + '_' + str(x) for x in value_sets[variable][1:]]
            train_data_onehot[new_columns] = [dummy_rules[variable][x] for x in train_data[variable]]
            test_data_onehot[new_columns] = [dummy_rules[variable][x] for x in test_data[variable]]
    
    train_data_onehot[y] = train_data[y]
    test_data_onehot = torch.tensor(test_data_onehot.values, dtype = torch.float32)
    
#==============================================================================
    
    method_name = 'hnn'
    
    num_epochs = 1000
    batch_size = 64
    lr = 0.01
    train_prop = 0.8
    num_iter = 2
    
    in_dim = len(var_cont) + len(var_cate)
    dims = [8, 4]
    hidden_dims = list(np.array(dims) * in_dim)
    
    model_mu, model_var, _, _ = fit_hnn(train_data_onehot, y, num_iter, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed)
    
    with torch.no_grad():
        mu_hnn = model_mu(test_data_onehot).detach().numpy().squeeze()
        var_hnn = model_var(test_data_onehot).detach().numpy().squeeze()
    
    quantiles = gen_quantiles(mu_hnn, var_hnn, tau)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,hidden_dims,ECE,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},\"{}\",{},{}\n".format(set_seed, str(dims), cal_ECE(test_data[y], quantiles, tau), rmse(mu_hnn, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame([mu_hnn.tolist(), var_hnn.tolist(), test_data[y].tolist()]).T
    results.columns = ['mean', 'variance', 'true']
    results.to_csv(data_dir + method_name + "/" + "{}_{}.csv".format(set_seed, str(dims)), index = False)
    
#==============================================================================
    
    method_name = 'hnn_ensemble'
    
    num_epochs = 1000
    batch_size = 64
    lr = 0.01
    train_prop = 0.8
    num_iter = 2
    nn_Num = 5
    
    in_dim = len(var_cont) + len(var_cate)
    dims = [8, 4]
    hidden_dims = list(np.array(dims) * in_dim)
    
    mu_ensem = np.zeros((len(test_data), nn_Num))
    var_ensem = np.zeros((len(test_data), nn_Num))
    mu_ensem[:, 0] = mu_hnn
    var_ensem[:, 0] = var_hnn
    
    for n in range(1, nn_Num):
        model_mu, model_var, _, _ = fit_hnn(train_data_onehot, y, num_iter, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed + n * 1997)
        
        with torch.no_grad():
            mu_ensem[:, n] = model_mu(test_data_onehot).detach().numpy().squeeze()
            var_ensem[:, n] = model_var(test_data_onehot).detach().numpy().squeeze()
    
    results = pd.DataFrame(mu_ensem, copy = True)
    results.columns = ['mean'+str(x) for x in range(nn_Num)]
    results[['variance'+str(x) for x in range(nn_Num)]] = var_ensem
    
    var_ensem2 = var_ensem.mean(axis = 1)
    var_ensem = (var_ensem + mu_ensem ** 2).mean(axis = 1) - mu_ensem.mean(axis = 1) ** 2
    mu_ensem = mu_ensem.mean(axis = 1)
    quantiles = gen_quantiles(mu_ensem, var_ensem, tau)
    quantiles2 = gen_quantiles(mu_ensem, var_ensem2, tau)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,hidden_dims,nn_Num,ECE,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},\"{}\",{},{},{}\n".format(set_seed, str(dims), nn_Num, 
            cal_ECE(test_data[y], quantiles, tau), rmse(mu_ensem, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results['mean'] = mu_ensem
    results['variance'] = var_ensem
    results['variance_a'] = var_ensem2
    results['true'] = test_data[y].tolist()
    results.to_csv(data_dir + method_name + "/" + "{}_{}_{}.csv".format(set_seed, str(dims), nn_Num), index = False)

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
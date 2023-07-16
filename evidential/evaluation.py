# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

from evidential import *

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
    
    method_name = 'evidential'
    
    num_epochs = 1000
    batch_size = 64
    lr = 0.01
    train_prop = 0.8
    Lambda = 0.05
    
    in_dim = len(var_cont) + len(var_cate)
    dims = [8, 4]
    hidden_dims = list(np.array(dims) * in_dim)
    
    model = fit_evidential(train_data_onehot, y, Lambda, hidden_dims, num_epochs, batch_size, lr, train_prop, set_seed)
    gamma, v, alpha, beta, nll_loss = predict_evidential(model, test_data_onehot, test_data[y])
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,hidden_dims,Lambda,NLL,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},\"{}\",{},{},{}\n".format(set_seed, str(dims), Lambda, nll_loss, rmse(gamma, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame([gamma.tolist(), v.tolist(), alpha.tolist(), beta.tolist(), test_data[y].tolist()]).T
    results.columns = ['gamma', 'v', 'alpha', 'beta', 'true']
    results.to_csv(data_dir + method_name + "/" + "{}_{}_{}.csv".format(set_seed, str(dims), Lambda), index = False)

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

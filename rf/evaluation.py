# -*- coding: utf-8 -*-

import os
import copy
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

from skopt.learning import RandomForestRegressor

def rmse(x, y):
    x_ = np.array(x).reshape(-1)
    y_ = np.array(y).reshape(-1)
    return np.sqrt(((x_ - y_) ** 2).mean())

# Negative log-likelihood loss
def NLLloss(y, mean, var):
    y_ = np.array(y).reshape(-1)
    mean_ = np.array(mean).reshape(-1)
    var_ = np.array(var).reshape(-1)
    return (np.log(var_) + (y_ - mean_) ** 2 / var_).mean()

#==============================================================================

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_ECE(true, pred, tau):
    ece = np.mean(np.abs(np.mean(np.array(true).reshape(-1, 1) < pred, axis = 0) - tau.reshape(-1)))
    return ece

#==============================================================================

def calibration(data_name, train_data, valid_data, test_data, var_cont, var_cate, y, set_seed):
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
    valid_data_onehot = pd.DataFrame(valid_data[list(var_cont)], copy = True)
    test_data_onehot = pd.DataFrame(test_data[list(var_cont)], copy = True)
    
    for variable in var_cate:
        if variable in dummy_rules:
            new_columns = [str(variable) + '_' + str(x) for x in value_sets[variable][1:]]
            train_data_onehot[new_columns] = [dummy_rules[variable][x] for x in train_data[variable]]
            valid_data_onehot[new_columns] = [dummy_rules[variable][x] for x in valid_data[variable]]
            test_data_onehot[new_columns] = [dummy_rules[variable][x] for x in test_data[variable]]
    
#==============================================================================
    
    method_name = 'rf'
    
    Trees = [50, 100, 150, 200]
    Depths = [4, 6, 8, 10, 12]
    Features = [0.3, 0.5, 0.7, 0.9]
    
    best_valid = 1e8
    for t in Trees:
        for d in Depths:
            for f in Features:
                rf = RandomForestRegressor(n_estimators = t, max_depth = d, max_features = f, 
                                           n_jobs = 16, random_state = set_seed)
                model_rf = rf.fit(train_data_onehot, train_data[y])
                mu_rf, std_rf = model_rf.predict(valid_data_onehot, return_std = True)
                valid_loss = NLLloss(valid_data[y], mu_rf, std_rf ** 2)
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    best_model = copy.deepcopy(model_rf)

    mu_rf, std_rf = best_model.predict(test_data_onehot, return_std = True)
    var_rf = std_rf ** 2
    quantiles = gen_quantiles(mu_rf, var_rf, tau)
    
    if not os.path.isfile(data_dir + method_name + ".csv"):
        with open(data_dir + method_name + ".csv", 'a') as file:
            file.write('train_test_seed,ECE,rmse\n')
    with open(data_dir + method_name + ".csv", 'a') as file:
        file.write("{},{},{}\n".format(set_seed, cal_ECE(test_data[y], quantiles, tau), rmse(mu_rf, test_data[y])))
    
    if not os.path.isdir(data_dir + method_name + "/"):
        os.mkdir(data_dir + method_name + "/")
    results = pd.DataFrame([mu_rf.tolist(), var_rf.tolist(), test_data[y].tolist()]).T
    results.columns = ['mean', 'variance', 'true']
    results.to_csv(data_dir + method_name + "/" + "{}.csv".format(set_seed), index = False)

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
        
        Train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        valid_index = random.sample(range(len(Train_data)), math.floor(len(Train_data) * (1-Train_proportion)))
        train_index = list(set(range(len(Train_data))) - set(valid_index))
        valid_data = Train_data.iloc[valid_index]
        train_data = Train_data.iloc[train_index]
        
        calibration(data_name, train_data, valid_data, test_data, var_cont, var_cate, y, set_seed)
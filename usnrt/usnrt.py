# -*- coding: utf-8 -*-
"""
Authors: Wenxuan Ma & Xing Yan @ RUC
mawenxuan@ruc.edu.cn
xingyan@ruc.edu.cn
"""

import random
import numpy as np
import pandas as pd
from scipy.stats import levene

from fit_nn import *
from fit_hnn import *

#==============================================================================
# Tree Structure

class Tree(object):
    def __init__(self, num_points, index, region_info):
        
        self.num_points = num_points
        self.index = index
        self.region_info = region_info
        
        self.dummy_rules = None
        
        self.model = None
        self.residual = None
        self.sigma2 = None
        
        self.model_uncertainty = None
        self.var_hat = None
        
        self.split_variable = None
        self.split_value = None
        self.left = None
        self.right = None
    
    def getSigma2(self):
        if self.left == None:
            return [self.sigma2]
        return self.left.getSigma2() + self.right.getSigma2()
    
    def getNumPoints(self):
        if self.left == None:
            return [self.num_points]
        return self.left.getNumPoints() + self.right.getNumPoints()
    
    def predict(self, x, var_cont, var_cate):
        if self.left == None:
            x_ = x[var_cont].iloc[0].tolist()
            for variable in var_cate:
                if variable in self.dummy_rules:
                    try:
                        x_ += self.dummy_rules[variable][x[variable].iloc[0]]
                    except: # Extremely unusual case when x[variable].iloc[0] is not in the dict
                        tmp = list(self.dummy_rules[variable].items())[0][1]
                        x_ += [0.0] * len(tmp)
            x_ = torch.tensor(x_, dtype = torch.float32).reshape(1, -1)
            predict_value = self.model(x_).detach().numpy()
            return predict_value.item()
        
        if self.split_variable in var_cont:
            if x[self.split_variable].iloc[0] <= self.split_value:
                return self.left.predict(x, var_cont, var_cate)
            else:
                return self.right.predict(x, var_cont, var_cate)
        else:
            if x[self.split_variable].iloc[0] in self.split_value[0]:
                return self.left.predict(x, var_cont, var_cate)
            else:
                return self.right.predict(x, var_cont, var_cate)
    
    def predict_all(self, data, var_cont, var_cate):
        y_hat = []
        for i in range(len(data)):
            x = data.iloc[[i]]
            y_hat.append(self.predict(x, var_cont, var_cate))
        return y_hat
    
    def sigma2(self, x, var_cont, var_cate):
        if self.left == None:
            return self.sigma2
        if self.split_variable in var_cont:
            if x[self.split_variable].iloc[0] <= self.split_value:
                return self.left.sigma2(x, var_cont, var_cate)
            else:
                return self.right.sigma2(x, var_cont, var_cate)
        else:
            if x[self.split_variable].iloc[0] in self.split_value[0]:
                return self.left.sigma2(x, var_cont, var_cate)
            else:
                return self.right.sigma2(x, var_cont, var_cate)
    
    def sigma2_all(self, data, var_cont, var_cate):
        sig_all = []
        for i in range(len(data)):
            x = data.iloc[[i]]
            sig_all.append(self.sigma2(x, var_cont, var_cate))
        return sig_all
    
    def uncertainty(self, x, var_cont, var_cate):
        if self.left == None:
            x_ = x[var_cont].iloc[0].tolist()
            for variable in var_cate:
                if variable in self.dummy_rules:
                    try:
                        x_ += self.dummy_rules[variable][x[variable].iloc[0]]
                    except: # Extremely unusual case when x[variable].iloc[0] is not in the dict
                        tmp = list(self.dummy_rules[variable].items())[0][1]
                        x_ += [0.0] * len(tmp)
            x_ = torch.tensor(x_, dtype = torch.float32).reshape(1, -1)
            variance = self.model_uncertainty(x_).detach().numpy()
            return variance.item()
        
        if self.split_variable in var_cont:
            if x[self.split_variable].iloc[0] <= self.split_value:
                return self.left.uncertainty(x, var_cont, var_cate)
            else:
                return self.right.uncertainty(x, var_cont, var_cate)
        else:
            if x[self.split_variable].iloc[0] in self.split_value[0]:
                return self.left.uncertainty(x, var_cont, var_cate)
            else:
                return self.right.uncertainty(x, var_cont, var_cate)
    
    def uncertainty_all(self, data, var_cont, var_cate):
        variances = []
        for i in range(len(data)):
            x = data.iloc[[i]]
            variances.append(self.uncertainty(x, var_cont, var_cate))
        return variances

#==============================================================================
# One-hot Data Generation

def get_onehot(data, var_cont, var_cate, y):
    value_sets = {}
    dummy_rules = {}
    for variable in var_cate:
        value_set = sorted(list(set(data[variable])))
        if len(value_set) <= 1 or len(value_set) > 10:
            continue
        value_sets[variable] = value_set
        dummies = np.identity(len(value_set))[:, 1:].tolist()
        dummy_rules[variable] = dict(zip(value_set, dummies))
    
    data_onehot = pd.DataFrame(data[list(var_cont)], copy = True)
    for variable in var_cate:
        if variable in dummy_rules:
            new_columns = [str(variable) + '_' + str(x) for x in value_sets[variable][1:]]
            data_onehot[new_columns] = [dummy_rules[variable][x] for x in data[variable]]
    data_onehot[y] = data[y]
    return dummy_rules, data_onehot

#==============================================================================
# Growing Tree

def grow_tree(data, var_cont, var_cate, y, split_dims, leaf_dims, p_value = 0.05,
              Nmin = 1000, split_num = 10, num_epochs = 1000, batch_size = 64,
              lr = 0.01, train_prop = 0.8, set_seed = 42, index = [1], region_info = 'All Data'):
    
    dummy_rules, data_onehot = get_onehot(data, var_cont, var_cate, y)
    root = Tree(num_points = len(data), index = index[0], region_info = region_info)
    root.dummy_rules = dummy_rules
    
    print('')
    print('root', index[0])
    
    if len(data) < 2*Nmin:
        root.model, root.model_uncertainty, Y_hat, root.var_hat = fit_hnn(data_onehot, 
            y, 2, leaf_dims, num_epochs, batch_size, lr, train_prop, set_seed + index[0])
        root.residual = data_onehot[y].values.reshape(-1) - Y_hat
        root.sigma2 = np.mean(root.residual ** 2)
        return root
    
    root.model, root.residual, root.sigma2 = fit_nn(data_onehot, var_cont, var_cate, y,
                split_dims, num_epochs, batch_size, lr, train_prop, set_seed + index[0])
    split_variable, split_value = get_best_split(data, root.residual, var_cont,
                    var_cate, y, p_value, Nmin, split_num, set_seed + index[0])
    
    if split_value == None:
        root.model, root.model_uncertainty, Y_hat, root.var_hat = fit_hnn(data_onehot, 
            y, 2, leaf_dims, num_epochs, batch_size, lr, train_prop, set_seed + index[0])
        root.residual = data_onehot[y].values.reshape(-1) - Y_hat
        root.sigma2 = np.mean(root.residual ** 2)
        return root
    
    root.split_variable = split_variable
    root.split_value = split_value
    
    if split_variable in var_cont:
        data_left = data[data[split_variable] <= split_value]
        data_right = data[data[split_variable] > split_value]
        region_info_left = str(split_variable) + '<=' + str(round(split_value, 4))
        region_info_right = str(split_variable) + '>' + str(round(split_value, 4))
    else:
        data_left = data[data[split_variable].isin(split_value[0])]
        data_right = data[data[split_variable].isin(split_value[1])]
        region_info_left = str(split_variable) + ':' + str(split_value[0])
        region_info_right = str(split_variable) + ':' + str(split_value[1])
    
    index[0] = index[0] + 1
    root.left = grow_tree(data_left, var_cont, var_cate, y, split_dims, leaf_dims,
                           p_value, Nmin, split_num, num_epochs, batch_size, lr, 
                           train_prop, set_seed, index, region_info_left)
    
    index[0] = index[0] + 1
    root.right = grow_tree(data_right, var_cont, var_cate, y, split_dims, leaf_dims,
                           p_value, Nmin, split_num, num_epochs, batch_size, lr, 
                           train_prop, set_seed, index, region_info_right)
    
    return root

#==============================================================================
# Split Criterion

def get_best_split(data, residual, var_cont, var_cate, y, p_value, Nmin, split_num, set_seed):
    best_split_variable = None
    best_split_value = None
    p_best = p_value
    
    data_new = pd.DataFrame(data, copy=True)
    data_new['residual'] = residual
    
    for variable in var_cate:
        split_set = generate_set(set(data_new[variable]), split_num, set_seed)
        for split_value in split_set:
            data1 = data_new[data_new[variable].isin(split_value[0])]
            data2 = data_new[data_new[variable].isin(split_value[1])]
            if len(data1) < Nmin or len(data2) < Nmin:
                continue
            
            stat, p = levene(data1['residual'], data2['residual'])
            if p < p_best:
                p_best = p
                best_split_value = split_value
                best_split_variable = variable
    
    for variable in var_cont:
        split_var = pd.DataFrame(data_new[[variable, 'residual']], copy = True).sort_values(variable)
        for i in range(Nmin, len(split_var) - Nmin + 1, 10):
            split_value = split_var[variable].iloc[i-1]
            if split_value == split_var[variable].iloc[i]:
                continue
            
            residual1 = split_var['residual'].iloc[:i]
            residual2 = split_var['residual'].iloc[i:]
            stat, p = levene(residual1, residual2)
            if p < p_best:
                p_best = p
                best_split_value = split_value
                best_split_variable = variable
    
    # print("p-value: {}, {}: {}".format(p_best, best_split_variable, best_split_value))
    return best_split_variable, best_split_value

#==============================================================================
# Split Criterion Helper

def generate_set(split_var, split_num, set_seed):
    if len(split_var) > split_num:
        random.seed(set_seed)
        subset_var = random.sample(split_var, split_num)
        rest_var = split_var - set(subset_var)
        rest_var0 = random.sample(rest_var, len(rest_var) // 2)
        rest_var0 = set(rest_var0)
        rest_var1 = rest_var - rest_var0
        split_set = list(pairs(subset_var))
        split_set = [[(x[0] | rest_var0), (x[1] | rest_var1)] for x in split_set]
    else:
        split_set = list(pairs(list(split_var)))
    return split_set

def pairs(split_var):
    n = len(split_var)
    test_marks = [1 << i for i in range(0, n)]
    pair_1 = list(range(1, 2**n-1)[0::2])
    pair_2 = list(range(1, 2**n-1)[1::2])
    pair_2.reverse()
    pair = list(zip(pair_1, pair_2))
    for k1, k2 in pair:
        l1 = []
        l2 = []
        for idx, item in enumerate(test_marks):
            if k1 & item:
                l1.append(split_var[idx])
            if k2 & item:
                l2.append(split_var[idx])
        yield [set(l1), set(l2)]
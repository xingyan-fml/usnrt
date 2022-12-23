# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

import usnrt_short as usnrt

#==============================================================================

def plot_split(data_name, data, var_cont, var_cate, y):
    
    if not os.path.isdir("results/"):
        os.mkdir("results/")
    data_dir = "results/"
    print('')
    print(data_name)
    print(len(data), len(var_cont) + len(var_cate))
    
    Train_proportion = 0.8 if len(data) < 3e5 else 0.5
    set_seed = 42
    
    data[var_cont] = (data[var_cont] - data[var_cont].mean()) / data[var_cont].std()
    data[y] = (data[y] - data[y].mean()) / data[y].std()
        
    random.seed(set_seed)
    train_index = random.sample(range(len(data)), math.floor(len(data) * Train_proportion))
    train_data = data.iloc[train_index]
    
#==============================================================================
# Our Tree
    
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
    
    if our_tree.split_variable not in var_cont:
        return
    for i in range(min(len(var_cont), 20)):
        if our_tree.split_variable == var_cont[i]:
            continue
        else:
            var = var_cont[i]
        
        data_split = pd.DataFrame([our_tree.residual**2, train_data[our_tree.split_variable], train_data[var]]).T
        data_split.columns = ['residual2', 'split_variable', var]
        quantile_transformer = preprocessing.QuantileTransformer(random_state = 42)
        data_split['residual2_uniform'] = quantile_transformer.fit_transform(data_split[['residual2']])
        
        data_split_left = data_split[data_split['split_variable'] <= our_tree.split_value]
        data_split_right = data_split[data_split['split_variable'] > our_tree.split_value]
        sigma2_left = round(data_split_left['residual2'].mean()*100, 2)
        sigma2_right = round(data_split_right['residual2'].mean()*100, 2)
        
        plt.figure(figsize=(6, 4),dpi = 300)
        fig = sns.scatterplot(data=data_split, x='split_variable', y=var, hue="residual2_uniform")
        plt.vlines(our_tree.split_value, plt.ylim()[0], plt.ylim()[1], color="black", linestyles='dashed', linewidth = 2)
        plt.xlabel('split variable: '+our_tree.split_variable)
        plt.ylabel('other variable: '+var)
        first_legend = plt.legend(title='residual^2', loc='lower right')
        plt.gca().add_artist(first_legend)
        
        h1, = plt.plot([], [], ' ', label="left $\\sigma_{{\\varepsilon}}^2$: {}".format(sigma2_left))
        h2, = plt.plot([], [], ' ', label="right $\\sigma_{{\\varepsilon}}^2$: {}".format(sigma2_right))
        plt.legend(handles=[h1,h2], title='region-specific $\\sigma_{{\\varepsilon}}^2$ (%)', loc='upper right')
        
        # x1 = (our_tree.split_value+data_split[our_tree.split_variable].min())/2
        # x2 = (our_tree.split_value+data_split[our_tree.split_variable].max())/2
        # y = data_split[var].max()
        # s1 = str(our_tree.split_variable)+'<='+str(round(our_tree.split_value, 4))+'\n sigma2={}'.format(sigma2[0])
        # s2 = str(our_tree.split_variable)+'>'+str(round(our_tree.split_value, 4))+'\n sigma2={}'.format(sigma2[1])
        # plt.text(x1, y, s1, color="black", fontsize = 8, verticalalignment = 'center')
        # plt.text(x2, y, s2, color="black", fontsize = 8, verticalalignment = 'center')
        
        fig_path = data_dir + data_name + '_' + str(i) + '.png'
        scatter_fig = fig.get_figure()
        scatter_fig.savefig(fig_path, bbox_inches='tight')
        plt.close()
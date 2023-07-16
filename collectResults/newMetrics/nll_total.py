# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

#==============================================================================

tau = np.array(range(1, 100)) / 100

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_ECE(true, pred, tau):
    ece = np.mean(np.abs(np.mean(np.array(true).reshape(-1, 1) < pred, axis = 0) - tau.reshape(-1)))
    return ece

def nll(true, mu, var):
    true_ = np.array(true).reshape(-1)
    mu_ = np.array(mu).reshape(-1)
    var_ = np.maximum(np.array(var).reshape(-1), 1e-8)
    nll = np.log(var_ * 2 * np.pi) + (true_ - mu_) ** 2 / var_
    return 0.5 * nll.mean()

def null_ensemble(true, mus, vars):
    density = pd.Series([0.] * true.size, index = true.index)
    for i in range(len(mus)):
        varsi = np.maximum(vars[i], 1e-8)
        density += np.exp(- 0.5 * (true - mus[i]) ** 2 / varsi) / np.sqrt(2 * np.pi * varsi)
    density = np.maximum(density / len(mus), 1e-16)
    return - np.mean(np.log(density))

#==============================================================================

data_list = ['Electrical','Conditional','Appliances','Real-time','Industry','Facebook1','Beijing',
             'Physicochemical','Traffic','Blog','Power','Online','Facebook2','Year','Query','GPU','Wave']
data_dirs = ["Electrical Grid Stability","Conditional Based Maintenance","Appliances energy prediction",
             "Real-time Election","Industry Energy Consumption","facebook1","Beijing PM2.5",
             "Physicochemical Properties","Traffic Volume","blog","Power consumption of T",
             "Online Video","facebook2","year","Query Analytics","GPU kernel performance","wave"]
data_dirs = dict(zip(data_list,data_dirs))

methods = ["mc_dropout","concrete_dropout","hnn_ensemble",'evidential','usnrt_ensemble']
methods_disp = ["\\makecell{MC\\\\Dropout}","\\makecell{Concrete\\\\Dropout}",
                "\\makecell{Deep\\\\Ensemble}",'Evidential','\\makecell{USNRT\\\\Ensemble}']
methods_disp = dict(zip(methods,methods_disp))

#==============================================================================

all_metrics = []
for data_name in data_list:
    metrics = []
    
    for method in methods:
        if method == "hnn_ensemble":
            resultDir = "../../{}/results/{}/{}/".format('hnn',data_dirs[data_name],method)
        elif method == "usnrt_ensemble":
            resultDir = "../../{}/results/{}/{}/".format('usnrt',data_dirs[data_name],method)
        else:
            resultDir = "../../{}/results/{}/{}/".format(method,data_dirs[data_name],method)
        fileList = os.listdir(resultDir)
        if method == 'mc_dropout':
            fileList = [x for x in fileList if '_0.5.csv' in x]
        if method == 'evidential':
            fileList = [x for x in fileList if '_0.01.csv' in x]
        
        if method == 'evidential':
            results = pd.read_csv(resultDir[:-1] + '.csv')
            results = results[results['Lambda'] == 0.01]
            per = results['NLL'].mean()
        elif method in ["hnn_ensemble", "usnrt_ensemble"]:
            per = []
            for fileName in fileList:
                results = pd.read_csv(resultDir+fileName)
                per.append(null_ensemble(results['true'], [results['mean' + str(x)] for x in range(5)], 
                                         [results['variance' + str(x)] for x in range(5)]))
            per = np.mean(per)
        else:
            per = []
            for fileName in fileList:
                results = pd.read_csv(resultDir+fileName)
                per.append(nll(results['true'], results['mean'], results['variance']))
            per = np.mean(per)
        
        metrics.append(per)
    all_metrics.append(pd.DataFrame([metrics], columns = methods))

all_metrics = pd.concat(all_metrics, ignore_index = True)
all_metrics.index = data_list

#==============================================================================

with open("nll_total.txt", 'w') as file:
    file.write("Dataset $\\backslash$ Method")
    for method in all_metrics.columns:
        file.write(" & "+methods_disp[method])
    # file.write(" & \\makecell{Percentage\\\\Decrease}")
    file.write(" \\\\ \\midrule\n")
    for data_name in data_list:
        metrics = all_metrics.loc[data_name]
        idxmin = metrics.sort_values().index[0]
        idxmin2 = metrics.sort_values().index[1]
        file.write(data_name)
        
        for method in metrics.index:
            if method == idxmin:
                # file.write(" & $^{{**}}${:.2f}".format(metrics[method]*100))
                file.write(" & \\textbf{{{:.2f}}}".format(metrics[method]*100))
            elif method == idxmin2:
                file.write(" & $^*${:.2f}".format(metrics[method]*100))
            else:
                file.write(" & {:.2f}".format(metrics[method]*100))
        
        # excp = metrics.drop('usnrt_ensemble').min()
        # reduceRate = (excp - metrics['usnrt_ensemble']) / excp
        # file.write(" & {:.1f}\\%".format(reduceRate*100))
        
        if data_name == data_list[-1]:
            file.write(" \\\\ \\bottomrule\n")
        else:
            file.write(" \\\\\n")

for data_name in data_list:
    metrics = all_metrics.loc[data_name]
    idxmin = metrics.sort_values().index[0]
    all_metrics.loc[data_name, idxmin] = '*' + str(all_metrics.loc[data_name, idxmin])

all_metrics.to_csv("nll_total.csv")



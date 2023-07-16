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

def rmse(true, pred):
    true_ = np.array(true).reshape(-1)
    pred_ = np.array(pred).reshape(-1)
    return np.sqrt(((true_ - pred_) ** 2).mean())

#==============================================================================

data_list = ['Electrical','Conditional','Appliances','Real-time','Industry','Facebook1','Beijing',
             'Physicochemical','Traffic','Blog','Power','Online','Facebook2','Year','Query','GPU','Wave']
data_dirs = ["Electrical Grid Stability","Conditional Based Maintenance","Appliances energy prediction",
             "Real-time Election","Industry Energy Consumption","facebook1","Beijing PM2.5",
             "Physicochemical Properties","Traffic Volume","blog","Power consumption of T",
             "Online Video","facebook2","year","Query Analytics","GPU kernel performance","wave"]
data_dirs = dict(zip(data_list,data_dirs))

methods = ['evidential0','evidential0.005','evidential0.01','evidential0.05','usnrt_ensemble']
methods_disp = ['\\makecell{Evidential\\\\$\\lambda=0$}','\\makecell{Evidential\\\\$\\lambda=0.005$}',
                '\\makecell{Evidential\\\\$\\lambda=0.01$}',
                '\\makecell{Evidential\\\\$\\lambda=0.05$}','\\makecell{USNRT\\\\Ensemble}']
methods_disp = dict(zip(methods,methods_disp))

#==============================================================================

all_metrics = []
for data_name in data_list:
    metrics = []
    
    for method in methods:
        if method == "usnrt_ensemble":
            resultDir = "../../{}/results/{}/{}/".format('usnrt',data_dirs[data_name],method)
        elif 'evidential' in method:
            resultDir = "../../{}/results/{}/{}/".format('evidential',data_dirs[data_name],'evidential')
        else:
            resultDir = "../../{}/results/{}/{}/".format(method,data_dirs[data_name],method)
        fileList = os.listdir(resultDir)
        
        if 'evidential' in method:
            Lambda = float(method[10:])
            results = pd.read_csv(resultDir[:-1] + '.csv')
            results = results[results['Lambda'] == Lambda]
            per = results['rmse'].mean()
        else:
            per = []
            for fileName in fileList:
                results = pd.read_csv(resultDir+fileName)
                per.append(rmse(results['true'], results['mean']))
            per = np.mean(per)
        
        metrics.append(per)
    all_metrics.append(pd.DataFrame([metrics], columns = methods))

all_metrics = pd.concat(all_metrics, ignore_index = True)
all_metrics.index = data_list

#==============================================================================

with open("rmse_evidential.txt", 'w') as file:
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

all_metrics.to_csv("rmse_evidential.csv")




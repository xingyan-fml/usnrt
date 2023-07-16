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

def cal_interval_ECE(true, pred, tau):
    true_s = np.array(true).reshape(-1,1)
    compare = np.logical_and(true_s > pred[:,[5-1,10-1,15-1,20-1]], true_s < pred[:,[-20,-15,-10,-5]][:,::-1])
    ece = np.mean(np.abs(np.mean(compare, axis = 0) - (1-tau.reshape(-1)[[5-1,10-1,15-1,20-1]]*2)))
    return ece

#==============================================================================

data_list = ['Electrical','Conditional','Appliances','Real-time','Industry','Facebook1','Beijing',
             'Physicochemical','Traffic','Blog','Power','Online','Facebook2','Year','Query','GPU','Wave']
data_dirs = ["Electrical Grid Stability","Conditional Based Maintenance","Appliances energy prediction",
             "Real-time Election","Industry Energy Consumption","facebook1","Beijing PM2.5",
             "Physicochemical Properties","Traffic Volume","blog","Power consumption of T",
             "Online Video","facebook2","year","Query Analytics","GPU kernel performance","wave"]
data_dirs = dict(zip(data_list,data_dirs))

methods = ["hnn","extra","rf",'usnrt']
methods_disp = ["HNN","\\makecell{Extra\\\\Trees}",
           "\\makecell{Random\\\\Forest}",'USNRT']
methods_disp = dict(zip(methods,methods_disp))

#==============================================================================

all_metrics = []
for data_name in data_list:
    metrics = []
    
    for method in methods:
        if method == "usnrt":
            resultDir = "../../{}/results/{}/{}/".format('usnrt',data_dirs[data_name],method)
        else:
            resultDir = "../../{}/results/{}/{}/".format(method,data_dirs[data_name],method)
        fileList = os.listdir(resultDir)
        if method == 'mc_dropout':
            fileList = [x for x in fileList if '_0.5.csv' in x]
        
        per = []
        for fileName in fileList:
            results = pd.read_csv(resultDir+fileName)
            quantiles = gen_quantiles(results['mean'].values, results['variance'].values, tau)
            per.append(cal_interval_ECE(results['true'], quantiles, tau))
        per = np.mean(per)
        
        metrics.append(per)
    all_metrics.append(pd.DataFrame([metrics], columns = methods))

all_metrics = pd.concat(all_metrics, ignore_index = True)
all_metrics.index = data_list

#==============================================================================

with open("tail_calibration_aleatory.txt", 'w') as file:
    file.write("Dataset $\\backslash$ Method")
    for method in all_metrics.columns:
        file.write(" & "+methods_disp[method])
    file.write(" & \\makecell{Percentage\\\\Decrease}")
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
        
        excp = metrics.drop('usnrt').min()
        reduceRate = (excp - metrics['usnrt']) / excp
        file.write(" & {:.1f}\\%".format(reduceRate*100))
        
        if data_name == data_list[-1]:
            file.write(" \\\\ \\bottomrule\n")
        else:
            file.write(" \\\\\n")

for data_name in data_list:
    metrics = all_metrics.loc[data_name]
    idxmin = metrics.sort_values().index[0]
    all_metrics.loc[data_name, idxmin] = '*' + str(all_metrics.loc[data_name, idxmin])

all_metrics.to_csv("tail_calibration_aleatory.csv")




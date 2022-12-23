# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#==============================================================================

tau = np.array(range(1, 100)) / 100

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_ECE(true, pred, tau):
    ece = np.mean(np.abs(np.mean(np.array(true).reshape(-1, 1) < pred, axis = 0) - tau.reshape(-1)))
    return ece

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

#==============================================================================

all_metrics = {'network_structure':{},'N_split':{}}
all_metrics['network_structure'] = {'calibration':{},'tail_calibration':{},'sharpness':{}}
all_metrics['N_split'] = {'calibration':{},'tail_calibration':{},'sharpness':{}}

for data_name in data_list:
    for variable in all_metrics.keys():
        resultDir = "../../usnrt_{}/results/{}/usnrt/".format(variable,data_dirs[data_name])
        fileList = os.listdir(resultDir)
        if variable == 'network_structure':
            parameters = ['_[4, 2]_[4, 2]_','_[8, 4]_[4, 2]_','_[16, 8]_[8, 4]_',
                          '_[8, 16, 4]_[4, 8, 2]_','_[16, 16, 8]_[8, 8, 4]_']
        else:
            parameters = ['_1.csv','_2.csv','_4.csv','_6.csv','_8.csv','_10.csv','_15.csv','_20.csv']
        
        for metric in ['calibration','tail_calibration','sharpness']:
            perList = []
            for param in parameters:
                param_fileList = [x for x in fileList if param in x]
                # print(data_name, variable, param, param_fileList)
                
                per = []
                for fileName in param_fileList:
                    results = pd.read_csv(resultDir+fileName)
                    quantiles = gen_quantiles(results['mean'].values, results['variance'].values, tau)
                    if metric == 'calibration':
                        per.append(cal_ECE(results['true'], quantiles, tau)*100)
                    elif metric == 'tail_calibration':
                        per.append(cal_interval_ECE(results['true'], quantiles, tau)*100)
                    else:
                        per.append(np.mean(np.sqrt(results['variance'].values))*100)
                per = np.mean(per)
                
                perList.append(per)
            all_metrics[variable][metric][data_name] = perList

#==============================================================================

linestyles = ['solid','dotted','dashed','dashdot','solid','dotted','dashed','dashdot','solid','dotted','dashed','dashdot','solid','dotted','dashed','dashdot','solid']
markers = ['o','v','^','s','*','+','x','D','1','X','d','2','8','P','>','p','<']
colors = ['b','g','r','c','m','y','k','gray','pink','lime','lightblue','orange','brown','purple','cyan','olive','black']

for variable in all_metrics.keys():
    for metric in ['calibration','tail_calibration','sharpness']:
        pdf = PdfPages("{}_{}.pdf".format(variable,metric))
        plt.figure(figsize=(7,4),dpi=300)
        
        i = 0
        for data_name in data_list:
            line = all_metrics[variable][metric][data_name]
            plt.plot(line, linestyle=linestyles[i], marker=markers[i], color=colors[i], label=data_name)
            i = i+1
        
        if variable == 'network_structure':
            plt.xticks(range(7),['S1','S2','S3','S4','S5','',''])
            plt.legend(loc='upper right',framealpha=0.5,fontsize=8)
        else:
            plt.xticks(range(11),['1','2','4','6','8','10','15','20','','',''])
            plt.legend(loc='upper right',framealpha=0.5,fontsize=8)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        pdf.close()
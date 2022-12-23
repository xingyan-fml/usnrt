# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#==============================================================================

tau = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def gen_quantiles(mu, variance, tau):
    quantiles = np.dot(np.sqrt(variance).reshape(-1, 1), norm.ppf(tau).reshape(1, -1))
    quantiles = quantiles + np.array(mu).reshape(-1, 1)
    return quantiles

def cal_calib_sharp(true, pred, tau):
    true_s = np.array(true).reshape(-1,1)
    compare = np.logical_and(true_s > pred[:,:tau.size//2], true_s < pred[:,-tau.size//2:][:,::-1])
    calib = np.mean(compare, axis = 0)# - (1-tau.reshape(-1)[:tau.size//2]*2)
    sharp = np.mean(pred[:,-tau.size//2:][:,::-1] - pred[:,:tau.size//2], axis = 0)
    return calib.tolist(), sharp.tolist()

#==============================================================================

data_list = ['Electrical','Conditional','Appliances','Real-time','Industry','Facebook1','Beijing',
             'Physicochemical','Traffic','Blog','Power','Online','Facebook2','Year','Query','GPU','Wave']
data_dirs = ["Electrical Grid Stability","Conditional Based Maintenance","Appliances energy prediction",
             "Real-time Election","Industry Energy Consumption","facebook1","Beijing PM2.5",
             "Physicochemical Properties","Traffic Volume","blog","Power consumption of T",
             "Online Video","facebook2","year","Query Analytics","GPU kernel performance","wave"]
data_dirs = dict(zip(data_list,data_dirs))

methods = ['usnrt',"hnn","hnn_ensemble","mc_dropout","concrete_dropout","extra","rf"]
methods_disp = ['USNRT',"HNN","Deep Ensemble","MC Dropout","Concrete Dropout","Extra Trees","Random Forest"]
methods_disp = dict(zip(methods,methods_disp))

#==============================================================================

all_metrics = {}
for data_name in data_list:
    metrics = {}
    
    for method in methods:
        if method == "hnn_ensemble":
            resultDir = "../../{}/results/{}/{}/".format('hnn',data_dirs[data_name],method)
        else:
            resultDir = "../../{}/results/{}/{}/".format(method,data_dirs[data_name],method)
        fileList = os.listdir(resultDir)
        if method == 'mc_dropout':
            fileList = [x for x in fileList if '_0.5.csv' in x]
        
        calib = []
        sharp = []
        for fileName in fileList:
            results = pd.read_csv(resultDir+fileName)
            quantiles = gen_quantiles(results['mean'].values, results['variance'].values, tau)
            per_calib, per_sharp = cal_calib_sharp(results['true'], quantiles, tau)
            calib.append(per_calib)
            sharp.append(per_sharp)
        calib = np.mean(np.array(calib), axis = 0)
        sharp = np.mean(np.array(sharp), axis = 0)
        
        metrics[method] = [calib, sharp]
    all_metrics[data_name] = metrics

#==============================================================================

linestyles = ['solid','dotted','dashed','dashdot','solid','dotted','dashed','dashdot']
markers = ['o','^','s','*','+','x','D','v']
colors = ['r','b','c','g','y','k','lime','gray']

for data_name in data_list:
    pdf = PdfPages("{}.pdf".format(data_name))
    plt.figure(figsize=(5.1, 3.4),dpi = 300)
    i = 0
    for method in methods:
        calib_sharp = all_metrics[data_name][method]
        calib, sharp = calib_sharp[0], calib_sharp[1]
        error = np.abs(calib - (1-tau.reshape(-1)[:tau.size//2]*2))[::-1]
        plt.plot(error, linestyle=linestyles[i], marker=markers[i], color=colors[i], label=methods_disp[method])
        i = i + 1
    plt.xticks(range(10),['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9',''])
    plt.legend(loc = 'upper right', framealpha = 0.5)
    pdf.savefig(bbox_inches='tight')
    plt.close()
    pdf.close()




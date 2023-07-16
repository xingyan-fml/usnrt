# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

markers = ['o','v','^','s','*','+','x','D','1','X','d','2','8','P','>','p','<']
colors = ['b','g','r','c','m','y','k','gray','pink','lime','lightblue','orange','brown','purple','cyan','olive','black']

data_list = ['Electrical','Conditional','Appliances','Real-time','Industry','Facebook1','Beijing',
             'Physicochemical','Traffic','Blog','Power','Online','Facebook2','Year','Query','GPU','Wave']
data_dirs = ["Electrical Grid Stability","Conditional Based Maintenance","Appliances energy prediction",
             "Real-time Election","Industry Energy Consumption","facebook1","Beijing PM2.5",
             "Physicochemical Properties","Traffic Volume","blog","Power consumption of T",
             "Online Video","facebook2","year","Query Analytics","GPU kernel performance","wave"]
data_dirs = dict(zip(data_list,data_dirs))

#==============================================================================

pdf = PdfPages("sigma_42.pdf")
plt.figure(figsize=(8,4),dpi=300)

i = 0
measures = []
for data_name in data_list:
    results = pd.read_csv("../../usnrt/results/" + data_dirs[data_name] + "/usnrt.csv")
    results = results[results['train_test_seed']==42]
    
    sigma2 = [float(x) for x in results.iloc[0]['sigma2'][1:-1].split(', ')]
    sigma = sorted(np.sqrt(sigma2))
    measures.append([data_name, np.std(np.log(sigma))])
    
    plt.scatter([i+1]*len(sigma), sigma, marker=markers[i], color=colors[i], label=data_name)
    i = i+1

plt.legend(framealpha=0.5,fontsize=8)
plt.xlim([0,21])
plt.ylim([0,1.5])
plt.xticks(range(1,18))
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()

#==============================================================================

measures = sorted(measures, reverse = True, key = lambda x: x[1])
with open("sigma_42.txt", 'w') as file:
    for elem in measures:
        file.write("{} & {:.2f} \\\\\n".format(elem[0], elem[1]))




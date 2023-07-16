# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

rates = []
with open('../metrics/calibration_aleatory.txt', 'r') as file: #
    for line in file:
        if 'Dataset' in line:
            continue
        data_name = line.split(' & ')[0]
        rate = float(line.split(' & ')[-1].split('\\')[0]) #
        rates.append([data_name, rate])

measures = []
with open('sigma_42.txt', 'r') as file:
    for line in file:
        data_name = line.split(' & ')[0]
        measure = float(line.split(' & ')[-1].split(' \\')[0]) #
        measures.append([data_name, measure])

rates_ = sorted(rates, key = lambda x:x[0])
measures_ = sorted(measures, key = lambda x:x[0])
print(spearmanr([x[1] for x in rates_], [x[1] for x in measures_])) #

#==============================================================================

with open("correlation.txt", 'w') as file:
    for elem in measures:
        for key in rates:
            if key[0] == elem[0]:
                rate = key[1]
                break
        file.write("{} & {:.2f} & {:.2f}\\% \\\\\n".format(elem[0], elem[1], rate))
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:56:29 2020

@author: lfiorentini
"""

import pandas as pd
from os.path import join as path_join

class joint_bil_filter():
    def __init__(self, param_V, param_G):
        self.__p1__ = param_V
        self.__p2__ = param_G
    
    def less_thres(self, col, thres):
        l1 = list(self.__p1__[self.__p1__[col] <= thres]['country'])
        l2 = list(self.__p2__[self.__p1__[col] <= thres]['country'])
        return l1, l2, set(l1) & set(l2)

    def more_thres(self, col, thres):
        l1 = list(self.__p1__[self.__p1__[col] >= thres]['country'])
        l2 = list(self.__p2__[self.__p1__[col] >= thres]['country'])
        return l1, l2, set(l1) & set(l2)


output_dir = './output'

param_V = pd.read_csv(path_join(output_dir, 'Verhulst', 'param.csv'),
                      sep = ';')
param_G = pd.read_csv(path_join(output_dir, 'Gompertz', 'param.csv'),
                      sep = ';')
param_VG = pd.read_csv(path_join(output_dir, 'Mixed_VG', 'param.csv'),
                      sep = ';')

Filter = joint_bil_filter(param_V, param_VG)


l1, l2, s1 = Filter.more_thres('estimated_completion_cases', 0.95)
l3, l4, s2 = Filter.more_thres('estimated_completion_deads', 0.95)
l5, l6, s3 = Filter.less_thres('current_mortality_ratio', 0.1)
l7, l8, s4 = Filter.less_thres('predict_mortality_ratio', 0.1)
l9, l10, s5 = Filter.less_thres('predict_cases_over_population', 0.001)
l11, l12, s6 = Filter.less_thres('predict_deads_over_population', 0.0001)

world_V = param_V[param_V['country'] == 'World']
saved_W_V = world_V['predict_total_cases'] - world_V['predict_total_deads']
France_V = param_V[param_V['country'] == 'France']
saved_F_V = France_V['predict_total_cases'] - France_V['predict_total_deads']
world_G = param_G[param_G['country'] == 'World']
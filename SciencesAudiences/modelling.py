# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:08:56 2020

@author: lfiorentini
version 1.1
"""

import time
from os.path import join as path_join
from datetime import datetime
from covid_utilities import scatter_param, load_data, hour_format, stop_day
from param_dealer import Param_dealer
from param_utilities import confront_errors
from covid_optimization import Optimizer

"""
LOADING AND PREPROCESSING

"""

input_dir = './input'
output_dir = './output'
min_values = 10

Cases, Deads, NEWCases, NEWDeads, Population, max_cases_p, max_deads_p =\
    load_data(input_dir, output_dir, min_values)

last_day = datetime.strptime(Cases['date'][len(Cases)-1], hour_format)
Cases['date'] = Cases.index
Deads['date'] = Deads.index
NEWCases['date'] = Cases.index
NEWDeads['date'] = Deads.index

"""
FITTING PARAMETERS

"""

start_time = time.time() 

warning_list = []

new_cols = ['country', 'n_data', 'err_real_data']

VOpt = Optimizer('Verhulst', warning_list, new_cols, start_time)
Ver_cases_p, Ver_deads_p, warning_list, end_time = VOpt.both_param_tables(
    Cases, Deads)

GOpt = Optimizer('Gompertz', warning_list, new_cols, end_time)
Gom_cases_p, Gom_deads_p, warning_list, end_time = GOpt.both_param_tables(
    Cases, Deads)

MixedOpt = Optimizer('Mixed_VG', warning_list, new_cols, end_time)
Mixed_cases_p, Mixed_deads_p, warning_list, end_time = \
    MixedOpt.both_param_tables(Cases, Deads)

AlleeOpt = Optimizer('Allee', warning_list, new_cols, end_time)
Allee_cases_p, Allee_deads_p, warning_list, end_time =\
    AlleeOpt.both_param_tables(Cases, Deads, NEWCases, NEWDeads)
    
AVGOpt = Optimizer('AVG', warning_list, new_cols, end_time)
AVG_cases_p, AVG_deads_p, warning_list, end_time =\
    AVGOpt.both_param_tables(Cases, Deads, NEWCases, NEWDeads)
    
"""
scatter graphs of parameters
"""
scatter_param(Ver_cases_p, path_join(output_dir, 'Verhulst'), 'a_k_cases_p')
scatter_param(Ver_deads_p, path_join(output_dir, 'Verhulst'), 'a_k_deads_p')
scatter_param(Gom_cases_p, path_join(output_dir, 'Gompertz'), 'a_k_cases_p')
scatter_param(Gom_deads_p, path_join(output_dir, 'Gompertz'), 'a_k_deads_p')


"""
PREDICTIONS

"""

date_to_predict = stop_day
mort_thres_ratio = 2.0
top_contries = ['China', 'France', 'Germany', 'Italy', 'Spain',
                'United Kingdom', 'United States', 'World']
PD = Param_dealer(Cases, Deads, max_cases_p, max_deads_p, Population,
                  mort_thres_ratio, input_dir, output_dir, hour_format,
                  date_to_predict, top_contries)

Ver_cases_p, Ver_deads_p, param_V, unreal_V = \
    PD.deal_w_param(Ver_cases_p, Ver_deads_p, 'Verhulst')
Gom_cases_p, Gom_deads_p, param_G, unreal_G = \
    PD.deal_w_param(Gom_cases_p, Gom_deads_p, 'Gompertz')
Mixed_cases_p, Mixed_deads_p, param_VG, unreal_VG = \
    PD.deal_w_param(Mixed_cases_p, Mixed_deads_p, 'Mixed_VG')
Allee_cases_p, Allee_deads_p, param_A, unreal_A = \
    PD.deal_w_param(Allee_cases_p, Allee_deads_p, 'Allee')
AVG_cases_p, AVG_deads_p, param_AVG, unreal_AVG = \
    PD.deal_w_param(AVG_cases_p, AVG_deads_p, 'AVG')

res = confront_errors([param_V, param_G, param_VG, param_A, param_AVG], 
                      ['Verhulst', 'Gompertz', 'Mixed_VG', 'Allee', 'AVG'])
res2 = res[[col for col in res.columns if 'day_rel_err_cases' in col] + 
           ['country']]
res3 = res[[col for col in res.columns if 'day_rel_err_deads' in col] + 
           ['country']]
opt_time_5 = time.time() 
print("Preparing data time: %s seconds " % (opt_time_5 - end_time))
# res2['min'] = res2.country.apply(lambda x: res2[res2.country == x].drop('country', axis = 1).idxmin(axis = 1))
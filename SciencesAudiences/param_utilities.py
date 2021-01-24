# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:31:22 2020

@author: lfiorentini
version 1.1
"""

import pandas as pd
import numpy as np
from os.path import join as path_join
from model_error import Model_error
from covid_predictor import Covid_predictor

def transform_param(param, mort_thres_ratio):
    """
    This functions take as an input a dataframe of countries with at least the
    following columns ['predict_total_cases', 'predict_total_deads',
                       'population', 'total_n_cases', 'total_n_deads'] and add
    to the dataframe the columns ['predict_cases_over_population',
                                  'predict_deads_over_population',
                                  'predict_mortality_ratio',
                                  'current_mortality_ratio'] removing countries
    with unrealistic data

    Parameters : 
    
    param : pd.DataFrame
        Input Dataframe.
    mort_thres_ratio : float
        maximum value admitted for the ratio betweene the expected mortality
        and the maximum one registered so far.

    Returns : 
    
    param : pd.DataFrame
        Dataframe processed.
    unreal_ls : list
        list of countries with unrealistic data.

    """
    param['predict_cases_over_population'] = param[
        'predict_total_cases']/ param['population']
    param['predict_deads_over_population'] = param[
        'predict_total_deads'] / param['population']

    param['predict_mortality_ratio'] = param['predict_total_deads'] / param[
        'predict_total_cases']

    param['current_mortality_ratio'] = param['total_n_deads'] / param[
        'total_n_cases']

    mort_thres = mort_thres_ratio * max(param['current_mortality_ratio'])
    unreal_ls = list(
        param[param['predict_mortality_ratio'] > mort_thres]['country'])
    param = param[param['predict_mortality_ratio'] <= mort_thres]

    unreal_ls += list(param[param['predict_total_cases'] > param['population']
                                ]['country'])
    param = param[param['predict_total_cases'] <= param['population']]

    unreal_ls += list(param[param['predict_total_deads'] > param['population']
                                ]['country'])
    param = param[param['predict_total_deads'] <= param['population']]

    param['estimated_completion_cases'] = param['total_n_cases'] / param[
        'predict_total_cases']
    param['estimated_completion_deads'] = param['total_n_deads'] / param[
        'predict_total_deads']

    return param, unreal_ls

def merge_datas(Cases_p, Deads_p, Population):
    """
    Merge the 3 dataframes

    Parameters :
    
    Cases_p : pd.DataFrame
        Cases_p dataframe.
    Deads_p : pd.DataFrame
        Deads_p dataframe.
    Population : pd.DataFrame
        Population dataframe.

    Returns :
    
    pd.DataFrame
        Merged dataframe.

    """
    Cases_p.columns = [c + '_cases' if c != 'country' else c
                     for c in Cases_p.columns]
    
    Deads_p.columns = [c + '_deads' if c != 'country' else c
                     for c in Deads_p.columns]
    
    return Cases_p.merge(Deads_p, on = 'country', how ='inner').merge(
        Population, on = 'country', how ='inner') 
    
def process_param(Cases, Deads, Cases_p, Deads_p, Population, mort_thres_ratio,
                  input_dir, output_dir, model):
    """
    This function process and save parameters tables

    Parameters :
    
    Cases : pd.DataFrame
        Cases dataframe.
    Deads : pd.DataFrame
        Deads dataframe.
    Cases_p : pd.DataFrame
        Cases params dataframe.
    Deads_p : pd.DataFrame
        Deads params dataframe.
    Population : pd.DataFrame
        Population dataframe.
    mort_thres_ratio : float
        threshold for the mortality ratio between the current and the predicted
        one.
    input_dir: str
        input directory.
    output_dir : str
        output directory.
    model : str
            str identifying the model used: possible values :
                {'Verhulst', 'Gompertz', 'Mixed_VG', 'Allee'}.

    Returns :
    
    param : pd.DataFrame
        Parameters dataframe.
    unreal_ls : list
        list of countries with unrealistic predictons.

    """
    Cases_p_error = Model_error(Cases_p, input_dir, output_dir, Cases,
                                model = model, label = 'Cases')
    Cases_p['stop_day_rel_err'] = Cases_p['country'].apply(lambda x: 
        Cases_p_error.stop_day_rel_err(x))
    
    Deads_p_error = Model_error(Deads_p, input_dir, output_dir, Deads,
                                model = model, label = 'Deads')
    Deads_p['stop_day_rel_err'] = Deads_p['country'].apply(lambda x: 
        Deads_p_error.stop_day_rel_err(x))

    Cases_p['predict_total'] = Cases_p['country'].apply(lambda x: 
        Covid_predictor(x, Cases_p[Cases_p['country'] == x], input_dir,
                        output_dir, Cases, model).predict_total())
    Deads_p['predict_total'] = Deads_p['country'].apply(lambda x: 
        Covid_predictor(x, Deads_p[Deads_p['country'] == x], input_dir,
                        output_dir, Deads, model).predict_total())
    
    param = merge_datas(Cases_p, Deads_p, Population)
    param, unreal_ls = transform_param(param, mort_thres_ratio)
    param.to_csv(path_join(output_dir, model, 'param.csv'), sep = ';',
                 index = False)
    outputfile = path_join(output_dir, model, 'unreal_ls.txt')
    with open(outputfile, 'w') as f:
        print('unreal_ls for', model, ':', unreal_ls, file = f)
        f.close()
    return param, unreal_ls

def confront_errors(list_param, list_models):
    """
    This function join a list of parameter dataframes on the errors

    Parameters :
    
    list_param : list
        list of parameters dataframes.
    list_models : list
        list of models name.

    Returns :
    
    result : pd.DataFrame
        dataframe with error comparison.

    """
    reduced_param = [el[['country', 'err_real_data_cases',
                         'err_real_data_deads', 'stop_day_rel_err_cases',
                         'stop_day_rel_err_deads']]
                     for el in list_param]
    
    result = pd.DataFrame(reduced_param[0]['country'])

    for i in range(len(reduced_param)):
        reduced_param[i].columns = ['country', list_models[i] + 'err_cases',
                                    list_models[i] + 'err_deads',
                                    list_models[i] + 'stop_day_rel_err_cases',
                                    list_models[i] + 'stop_day_rel_err_deads']
        result = result.merge(reduced_param[i], on = 'country', how ='outer')
    return result.fillna(np.inf)    
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:36:50 2020

@author: lfiorentini
version 1.1
"""

import pandas as pd
import time
from os.path import join as path_join
from covid_utilities import download_save


input_dir = './input'
output_dir = './output'
model = 'LSTM'
download_save(input_dir, 'total_cases')
download_save(input_dir, 'total_deaths')
download_save(input_dir, 'new_cases')
download_save(input_dir, 'new_deaths')
download_save(input_dir, 'locations')

Cases = pd.read_csv(path_join(input_dir, 'total_cases.csv'))
Cases.ffill(axis = 0, inplace = True)
Cases.fillna(0, inplace = True) 

Deads = pd.read_csv(path_join(input_dir, 'total_deaths.csv'))
Deads.ffill(axis = 0, inplace = True)
Deads.fillna(0, inplace = True)

Population = pd.read_csv(path_join(input_dir, 'locations.csv'))
Population.dropna(axis = 0, how = 'any', inplace = True)
Population['country'] = Population['location']
Population.drop(['countriesAndTerritories', 'population_year', 'location'],
                axis = 1, inplace = True)
Population = Population.append({'country' : 'World', 'continent' : 'World',
                   'population' : sum(Population.population)},
                  ignore_index = True)
Population.drop_duplicates(keep = 'first', inplace = True)

not_enough_data = []

for col in Cases.columns:
    if col not in Deads.columns:
        Cases.drop(col, axis = 1, inplace = True)

for col in Deads.columns:
    if col not in Cases.columns:
        Deads.drop(col, axis = 1, inplace = True)

for col in Cases.columns:
    if min(len(Cases[col].unique()), len(Deads[col].unique())) < min_val:
        if col not in not_enough_data:
            not_enough_data.append(col)
        
Cases.drop(not_enough_data, axis = 1, inplace = True)
Deads.drop(not_enough_data, axis = 1, inplace = True)
for col in Cases.columns:
    if col != 'date':
        max_col = Population[Population['country'] == col]['population']
        Cases[col] /= max_col
        Deads[col] /= max_col

terminated country = ['China', 'South Korea', 'Austria', 'Lebanon',
                      'New Zealand']


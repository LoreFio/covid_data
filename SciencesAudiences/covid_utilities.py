# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:11:48 2020

@author: lfiorentini
version 1.1
"""

import pandas as pd
import matplotlib.pyplot as plt
import requests
from os.path import join as path_join
from datetime import datetime, timedelta


hour_format = '%Y-%m-%d'
initial_day = datetime.strptime("2019-12-31", hour_format)
today = datetime.today()
yesterday = today - timedelta(1)
current_time = datetime.now()

if current_time.hour < 14:
    stop_day = datetime.strftime(yesterday, hour_format)
else:
    stop_day = datetime.strftime(today, hour_format)

def remove_day(data, day):
    """
    This function remove the data of one day from the dataset returning the
    removed line

    Parameters
    
    data : pd.DataFrame
        dataframe to be processes.
    day : str
        string of the day date.

    Returns
    
    stop_day_data : pd.DataFrame
        dataframe without the day.

    """
    stop_day_data = data[data['date'] == day]
    data['tmp'] = data['date'].apply(lambda x: datetime.strptime(x,
                                                                 hour_format))
    data = data[data['tmp'] < datetime.strptime(day, hour_format)]
    data.drop('tmp', axis = 1, inplace = True)
    return stop_day_data, data


def load_data(input_dir = './input', output_dir = './output', min_val = 6):
    """
    This functions downloads data, preprocess and return them 

    Parameters :
    
    input_dir : str, optional
        Input directory. The default is './input'.
    output_dir : str, optional
        Output directory. The default is './output'.
    min_val : int64, optional
        minimum number of data that a column must have. The default is 6.

    Returns : 
    
    Cases : pd.DataFrame
        DataFrame with the cases data.
    Deads : pd.DataFrame
        DataFrame with the deads data.
    Population : pd.DataFrame
        DataFrame with the population data.
    max_cases : pd.DataFrame
        DataFrame with the maximum cases data.
    max_deads : pd.DataFrame
        DataFrame with the maximum deads data.

    """
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

    NEWCases = pd.read_csv(path_join(input_dir, 'new_cases.csv'))
    NEWCases.ffill(axis = 0, inplace = True)
    NEWCases.fillna(0, inplace = True) 
    
    NEWDeads = pd.read_csv(path_join(input_dir, 'new_deaths.csv'))
    NEWDeads.ffill(axis = 0, inplace = True)
    NEWDeads.fillna(0, inplace = True)

    #remove stop_day values and put them in separate files
    stop_day_cases, Cases = remove_day(Cases, stop_day)    
    stop_day_deads, Deads = remove_day(Deads, stop_day)    
    stop_day_new_cases, NEWCases = remove_day(NEWCases, stop_day)
    stop_day_new_deads, NEWDeads = remove_day(NEWDeads, stop_day)

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
    NEWCases.drop(not_enough_data, axis = 1, inplace = True)
    NEWDeads.drop(not_enough_data, axis = 1, inplace = True)

    stop_day_cases.drop(not_enough_data, axis = 1, inplace = True)
    stop_day_deads.drop(not_enough_data, axis = 1, inplace = True)
    stop_day_new_cases.drop(not_enough_data, axis = 1, inplace = True)
    stop_day_new_deads.drop(not_enough_data, axis = 1, inplace = True)
    
    stop_day_cases.to_csv(path_join(input_dir, 'stop_day_cases.csv'), sep = ';',
                       index = False)
    stop_day_deads.to_csv(path_join(input_dir, 'stop_day_deads.csv'), sep = ';',
                       index = False)
    stop_day_new_cases.to_csv(path_join(input_dir, 'stop_day_new_cases.csv'),
                           sep = ';', index = False)
    stop_day_new_deads.to_csv(path_join(input_dir, 'stop_day_new_deads.csv'),
                           sep = ';', index = False)

    Population = pd.read_csv(path_join(input_dir, 'locations.csv'))
    Population.dropna(axis = 0, how = 'any', inplace = True)
    Population['country'] = Population['location']
    Population.drop(['countriesAndTerritories', 'population_year', 'location'],
                    axis = 1, inplace = True)
    Population = Population.append({'country' : 'World', 'continent' : 'World',
                       'population' : sum(Population.population)},
                      ignore_index = True)
    Population.drop_duplicates(keep = 'first', inplace = True)
    max_cases = pd.DataFrame(columns = ['country', 'total_n'])
    max_deads = pd.DataFrame(columns = ['country', 'total_n'])
    
    for col in Cases.columns:
        if col != 'date':
            max_c = max(Cases[col])
            max_d = max(Deads[col])
            Cases[col] /= max_c
            Deads[col] /= max_d
            NEWCases[col] /= max_c
            NEWDeads[col] /= max_d
            line1 = [col, max_c]
            line2 = [col, max_d]
            max_cases.loc[len(max_cases)] = line1
            max_deads.loc[len(max_deads)] = line2
        
    return Cases, Deads, NEWCases, NEWDeads, Population, max_cases, max_deads
    
def download_save(input_dir = './input', file = 'max_cases'):
    """
    This function download and save content from the website
    'https://covid.ourworldindata.org/data/ecdc/'

    Parameters
    
    input_dir : str, optional
        Input directory. The default is './input'.
    file : str, optional
        str identifying the file to download, possible values are
        {'max_cases', 'total_deaths', 'new_cases', 'new_deaths',
         'locations'}. The default is 'max_cases'.

    Returns
    
    None.

    """
    dest = file + '.csv'
    url = 'https://covid.ourworldindata.org/data/ecdc/' + dest
    myfile = requests.get(url)
    dest = path_join(input_dir, dest)
    with open(dest, 'wb') as f:
        f.write(myfile.content)
        f.close()

def first_day(data):
    """
    Return first occurence of a positive data in a column

    Parameters :
    
    data : pd.Series
        column to be studied.

    Returns :
    
    numpy.int64
        index of first position.

    """
    return data[data > 0].index[0]

def first_cases(data):
    """
    Return first positive value in a column

    Parameters :
    
    data : pd.Series
        column to be studied.

    Returns :
    
    numpy.int64
        first positive value.

    """
    return data[first_day(data)]

def plot_prediction(index, value, pred, output_dir, err = 0.0,
                     index_2 = None, name = "pred"):
    """
    Plot real and ext

    Parameters :
    
    index : pd.DataFrame
        indexes for the x axis.
    value : pd.DataFrame
        real value.
    pred : pd.DataFrame
        pred value.
    output_dir : str
        output directory.
    err : float64
        error of the prediction. The default is 0.0.
    index_2 : pd.DataFrame
        list like object with the index for the prediction.
        The default is None.
    name : str
        str with the name for the graph. The default is 'pred'.

    Returns :
    
    None.

    """
    if index_2 == None:
        index_2 = index
    plt.plot(index, value, color = 'b')
    plt.plot(index_2, pred, color = 'r')
    if err > 0.0:
        plt.plot(index_2, [p + err for p in pred], color = 'k')
        plt.plot(index_2, [p - err for p in pred], color = 'k')
        
    plt.legend(('real', 'estimation', 'upper bound', 'lower bound'))
    plt.title('Estimation ' + name)
    plt.savefig(path_join(output_dir, name) + ".png", bbox_inches='tight')
    plt.close()

def scatter_param(data, output_dir, name = 'a_k'):
    """
    Plot parameters of the logistic functions with country name on it

    Parameters :
    
    data : pd.DataFrame
        DataFrame with countries to be studied.
    output_dir : str
        output directory.
    name : str
        str with the name for the graph. The default is 'a_k'.

    Returns :
    
    None.

    """
    fig, ax = plt.subplots()
    ax.scatter(data['a'], data['k'])

    for i in range(len(data)):
        ax.annotate(data['country'][i], (data['a'][i], data['k'][i]))
    plt.savefig(path_join(output_dir, name) + ".png", bbox_inches='tight')
    plt.close()

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:14:45 2020

@author: lfiorentini
version 1.1
"""

import time
import warnings
import pandas as pd
from math import sqrt
from verhulst import Verhulst
from gompertz import Gompertz
from mixed_VG import Mixed_VG
from avg import Avg
from allee import Allee
from covid_utilities import first_day
from scipy.optimize import minimize

class Optimizer():
    def __init__(self, model, warning_list, new_cols, start_time):
        """
        Initializer of the optimizer

        Parameters :
        
        model : str
            str to identify the model used: possible values :
                {'Verhulst', 'Gompertz', 'Mixed_VG', 'Allee'}.
        warning_list : list
            list of countries for which warnings have been raised.
        new_cols : list
            list of the columns for the output dataframe.
        start_time : float
            time of the start
        Returns :
        
        None.

        """
        
        self.__model__ = model
        self.__warning_list__ = warning_list
        self.__new_cols__ = new_cols
        self.__start_time__ = start_time
        
    def param_table(self, data, data_der = None):
        """
        Solve the optimization problem of the model for all the countries
        returning a dataframe with the results of the simulation
    
        Parameters :
        
        data : pd.DataFrame/Series
            data to be used.
        data_der : pd.DataFrame
            DataFrame with the historical data derivative. The default is
            'None'.
    
        Returns :
        
        param_data : pd.DataFrame
            dataframe of parameters.

        """
        param_data = pd.DataFrame(columns = self.__new_cols__)
    
        for col in data.columns:
            if col != 'date' and col not in self.__warning_list__:
                col_len = len(data[col])
                useful_data = col_len - first_day(data[col])
                if self.__model__ == 'Verhulst':
                    mod = Verhulst(data[col])
                elif self.__model__ == 'Gompertz':
                    mod = Gompertz(data[col])
                elif self.__model__ == 'Mixed_VG':
                    mod = Mixed_VG(data[col])
                elif self.__model__ == 'Allee':
                    mod = Allee(data[col], data_der[col])
                elif self.__model__ == 'AVG':
                    mod = Avg(data[col], data_der[col])
                
                self.__param_cols__ = mod.param_cols
                self.__initial_p__ = mod.initial_p
                self.__model_bounds__ = mod.model_bounds
                
                #at the first time we have to add the other columns
                if len(param_data.columns) == len(self.__new_cols__):
                    param_data = pd.DataFrame(columns = self.__new_cols__ +
                                              self.__param_cols__)

                '''
                res = minimize(fun = mod.MSE, x0 = self.__initial_p__,
                               args = (data[col]), jac = grad_MSE,
                               method = 'L-BFGS-B',
                               bounds = self.__model_bounds__)
                '''
                with warnings.catch_warnings(record=True) as w:
                    
                    res = minimize(fun = mod.MSE, x0 = self.__initial_p__,
                                   method = 'L-BFGS-B',
                                   bounds = self.__model_bounds__)    
                    if len(w) > 0:
                        self.__warning_list__.append(col)
                    else:
                        err = sqrt(res.fun)
                        line = [col, useful_data,
                                err * sqrt( col_len / float(useful_data))]
                        for i in range(len(self.__param_cols__)):
                            line.append(res.x[i])
                        param_data.loc[len(param_data)] = line
                    if not res.success:
                        print(self.__model__, col, "unsuccess")
        return param_data

    def both_param_tables(self, Data_C, Data_D, der_C = None, der_D = None):
        """
        Solve the optimization problem of the model for all the countries
        and for both cases and deads returning a dataframe with the results of
        the simulation

        Parameters
        
        Data_C : pd.DataFrame
            dataframe of the cases.
        Data_D : pd.DataFrame
            dataframe of the deads.
        der_C : pd.DataFrame
            DataFrame with the historical data derivative for the cases.
            The default is 'None'.
        der_D : pd.DataFrame
            DataFrame with the historical data derivative for the deads.
            The default is 'None'.

        Returns
        
        param_C : pd.DataFrame
            dataframe of the cases parameters.
        param_D : pd.DataFrame
            dataframe of the deads parameters.
        list
            list of countries for which te computation provided errors.
        end_time : float
            time of the end

        """
        param_C = self.param_table(Data_C, der_C)
        param_D = self.param_table(Data_D, der_D)
        end_time  = time.time()
        print("Fitting", self.__model__, "time: %s seconds " % (
            end_time  - self.__start_time__))

        return param_C, param_D, self.__warning_list__, end_time
        
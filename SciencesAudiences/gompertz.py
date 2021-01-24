# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:08:08 2020

@author: lfiorentini
version 1.1
"""

import numpy as np
from math import exp, log
from covid_utilities import first_day, first_cases

class Gompertz():
    def __init__(self, data):
        """
        Initialiser

        Parameters :
        
        data : pd.Dataframe/Series
            Column of data of a country.

        Returns :
        
        None.

        """
        self.__data__ = data
        self.__to__ = first_day(data)
        self.__yo__ = first_cases(data)
        self.param_cols = ['a', 'k']
        self.initial_p = [0.1, 1]
        self.model_bounds = ((0, np.inf), (1, np.inf))

    def evolution(self, t, a, k, to, yo):
        """
        Evaluate the evolution function
    
        Parameters :
        
        t : float
            parameter of the evolution function t.
        a : float
            parameter of the evolution function a.
        k : float
            parameter of the evolution function k.
        to : float
            parameter of the evolution function to.
        yo : float
            parameter of the evolution function yo.
    
        Returns :
         
        float
            result.
    
        """
        if t < to:
            return 0
        return k * exp( log(yo / k) * exp( -a * (t - to) ) )
    
    def error_sq(self, t, a, k, to, y, yo):
        """
        Evaluate the evolution function
    
        Parameters :
        
        t : float
            parameter of the evolution function t.
        a : float
            parameter of the evolution function a.
        k : float
            parameter of the evolution function k.
        to : float
            parameter of the evolution function to.
        yo : float
            parameter of the evolution function yo.
        y : float
            real datum.
    
        Returns :
         
        float
            squared error.
    
        """
    
        return (y - self.evolution(t, a, k, to, yo)) ** 2
    
    def MSE(self, x):
        """
        Given the parameters of the evolution function in x computes the MSE over
        the data in the DataFrame
    
        Parameters :
        
        x : couple
            couple of a,k.
    
        Returns :
        
        float
            MSE.
    
        """
        a = x[0]
        k = x[1]
        res = 0
        n = len(self.__data__)
        for t in range(n):
            res += self.error_sq(t, a, k, self.__to__, self.__data__[t],
                            self.__yo__)
        return res / n
    
    def predict_max(self, line):
        """
        Given the parameters and the data it makes a prediction until a limit
    
        Parameters :
        
        line : pandas.core.frame.DataFrame
            Dataframe with the parameters.
        Returns :
        
        float
            limit that will be reached by the population.
    
        """
        return float(line['k'])
    
    def prediction_series(self, line, limit = None):
        """
        Given the parameters and the data it makes a prediction until a limit
    
        Parameters :
        
        line : pandas.core.frame.DataFrame
            Dataframe with the parameters.
        limit : int, optional
            number of days of the predicted curve. The default is None.
    
        Returns :
        
        list
            list with the values of the prediction from 0 to limit days.
    
        """
        a = float(line['a'])
        k = float(line['k'])
        if limit == None:
            limit = len(self.__data__)
        return [self.evolution(t, a, k, self.__to__, self.__yo__) 
                for t in range(limit)]

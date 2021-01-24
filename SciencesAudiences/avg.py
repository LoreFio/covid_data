# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:35:45 2020

@author: lfiorentini
version 1.1
"""

import numpy as np
from covid_utilities import first_day, first_cases
import scipy.integrate as spi
from math import log

class Avg():
    def __init__(self, data, data_der):
        self.__data__ = data
        self.__data_der__ = data_der
        self.__to__ = first_day(data)
        self.__yo__ = first_cases(data)
        self.param_cols = ['a', 'alpha', 'k', 'c']
        self.initial_p = [0.1, 0.1, 1, 0.5]
        self.model_bounds =  ((0, np.inf), (0, np.inf),
                              (1, np.inf), (-np.inf, np.inf))
     
    def formula(self, y, t, a, alpha, k, c):
        """
        Evaluate the formula
    
        Parameters :
        
        y : float
            parameter of the formula y.
        t : float
            parameter of the formula t.
        a : float
            parameter of the formula a.
        alpha : float
            parameter of the formula alpha.
        k : float
            parameter of the formula k.
        c : float
            parameter of the formula c.

        Returns :
         
        float
            result.
    
        """
        if t < self.__to__:
            return 0
        return a * y * c * (1 - y / k)  + alpha * y * (1 - c) * log(k/y)  
        
    def error_sq(self, y, to, t, a, alpha, k, c, y_p):
        """
        Evaluate the formula
    
        Parameters :
        
        y : float
            parameter of the formula y.
        to : float
            initial time to.
        t : float
            parameter of the formula t.
        a : float
            parameter of the formula a.
        alpha : float
            parameter of the formula alpha.
        k : float
            parameter of the formula k.
        c : float
            parameter of the formula c.
        y : float
            parameter of the formula y.
        y_p : float
            real datum derivative.
    
        Returns :
         
        float
            squared error.
    
        """
        if t < to:
            return (y_p)**2
        return (y_p - self.formula(y, t, a, alpha, k, c)) ** 2
                
    def MSE(self, x):
        """
        Given the parameters of the formula in x computes the MSE over
        the data in the DataFrame
    
        Parameters :
        
        x : list
            list of parameters.
    
        Returns :
        
        float
            MSE.
    
        """
        a = x[0]
        alpha = x[1]
        k = x[2]
        c = x[3]
        res = 0
        n = len(self.__data__)
        for t in range(n):
            res += self.error_sq(self.__data__[t], self.__to__, t, a, alpha, k, c,
                                 self.__data_der__[t])
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
        alpha = float(line['alpha'])
        k = float(line['k'])
        c = float(line['c'])
        if limit == None:
            limit = len(self.__data__)
        t = np.linspace(self.__to__, limit, limit - self.__to__ + 1)
        simulated = spi.odeint(self.formula, self.__yo__, t,
                               args = (a, alpha, k, c))
        before_y = np.zeros([self.__to__, 1])
        return list(np.concatenate((before_y, simulated), axis = 0))
        
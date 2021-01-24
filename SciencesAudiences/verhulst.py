# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:08:08 2020

@author: lfiorentini
version 1.1
"""

import numpy as np
from math import exp
from covid_utilities import first_day, first_cases

class Verhulst():
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

    def logistic(self, t, a, k, to, yo):
        """
        Evaluate the logistic function
    
        Parameters :
        
        t : float
            parameter of the logistic function t.
        a : float
            parameter of the logistic function a.
        k : float
            parameter of the logistic function k.
        to : float
            parameter of the logistic function to.
        yo : float
            parameter of the logistic function yo.
    
        Returns :
         
        float
            result.
    
        """
        if t < to:
            return 0
        return k / ( 1 + ( k / yo - 1) * exp( -a * (t - to) ) )
    
    def logistic_grad(self, t, a, k, to, yo):
        """
        Evaluate the logistic function gradient with respect to a, k
    
        Parameters :
        
        t : float
            parameter of the logistic function t.
        a : float
            parameter of the logistic function a.
        k : float
            parameter of the logistic function k.
        to : float
            parameter of the logistic function to.
        yo : float
            parameter of the logistic function yo.
    
        Returns :
         
        np.array()
            partial derivatives w.r.t a and k respectively.
    
        """
    
        if t < to:
            return [0, 0]
        der1 = self.logistic(t, a, k, to, yo) ** 2 / k * (t - to) * (
            k / yo - 1) * exp(-a * (t-to))
        der2 = yo * exp(a * (t - to)) * yo * (exp(a * (t - to)) - 1) / (
            (k + yo*(exp(a * (t - to)) - 1))**2)
        return np.array([der1, der2])
    
    def error_sq(self, t, a, k, to, y, yo):
        """
        Evaluate the logistic function
    
        Parameters :
        
        t : float
            parameter of the logistic function t.
        a : float
            parameter of the logistic function a.
        k : float
            parameter of the logistic function k.
        to : float
            parameter of the logistic function to.
        yo : float
            parameter of the logistic function yo.
        y : float
            real datum.
    
        Returns :
         
        float
            squared error.
    
        """
    
        return (y - self.logistic(t, a, k, to, yo)) ** 2
    
    def error_sq_grad(self, t, a, k, to, y, yo):
        """
        Evaluate the squared error gradient with respect to a, k
    
        Parameters :
        
        t : float
            parameter of the logistic function t.
        a : float
            parameter of the logistic function a.
        k : float
            parameter of the logistic function k.
        to : float
            parameter of the logistic function to.
        yo : float
            parameter of the logistic function yo.
        y : float
            real datum.
    
        Returns :
         
        np.array()
            partial derivatives w.r.t a and k respectively.
    
        """
    
        try :
            return 2 * (self.logistic(t, a, k, to, yo) - y) * np.array(
                self.logistic_grad(t, a, k, to, yo))
        except TypeError: 
            print(t, a, k, to, y, yo)
            print((self.logistic(t, a, k, to, yo) - y))
            print(self.logistic_grad(t, a, k, to, yo))
            print(type(self.logistic(t, a, k, to, yo) - y))
            print(type(self.logistic_grad(t, a, k, to, yo)))
            
    def MSE(self, x):
        """
        Given the parameters of the logistic function in x computes the MSE over
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
    
    def grad_MSE(self, x):
        """
        Given the parameters of the logistic function in x computes the gradient of
        the MSE over the data in the DataFrame with respect to a, k
    
        Parameters :
        
        x : couple
            couple of a,k.
    
        Returns :
        
        float
            MSE.
    
        """
        a = x[0]
        k = x[1]
        res = np.array([0, 0], dtype = np.float64)
        n = len(self.__data__)
        for t in range(n):
            partial = self.error_sq_grad(t, a, k, self.__to__,
                                         self.__data__[t], self.__yo__)
            res[0] += partial[0]
            res[1] += partial[1]
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
        return [self.logistic(t, a, k, self.__to__, self.__yo__) 
                for t in range(limit)]

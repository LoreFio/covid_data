# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:29:04 2020

@author: lfiorentini
version 1.1
"""

import numpy as np
from verhulst import Verhulst
from gompertz import Gompertz

class Mixed_VG():
    def __init__(self, data):
        """
        Initialiser

        Parameters :
        
        data : pd.Dataframe/Series
            Column of data of a country.

        Returns :
        
        None.

        """
        self.__v_model__ = Verhulst(data)
        self.__g_model__ = Gompertz(data)
        self.__data__ = self.__v_model__.__data__
        self.__to__ = self.__v_model__.__to__
        self.__yo__ = self.__v_model__.__yo__
        self.param_cols = ['a', 'alpha', 'k', 'c']
        self.initial_p = [0.1, 0.1, 1, 0.5]
        self.model_bounds =  ((0, np.inf), (0, np.inf),
                              (1, np.inf), (-np.inf, np.inf))

    def evolution(self, t, a, alpha, k, c, to, yo):
        """
        Evaluate the evolution function
    
        Parameters :
        
        t : float
            time parameter for both models.
        a : float
            parameter of the Verhulst model.
        alpha : float
            parameter of the evolution function alpha.
        k : float
            parameter for both models.
        c : float
            parameter of the evolution function c.
        to : float
            parameter of the evolution function to.
        yo : float
            parameter of the evolution function yo.
    
        Returns :
         
        float
            result.
    
        """

        return c * self.__v_model__.logistic(t, a, k, to, yo) +\
            (1 - c) * self.__g_model__.evolution(t, alpha, k, to, yo)
            
    def error_sq(self, t, a, alpha, k, c, to, y, yo):
        """
        Evaluate the evolution function
    
        Parameters :
        
        t : float
            parameter of the evolution function t.
        a : float
            parameter of the evolution function a.
        alpha : float
            parameter of the evolution function alpha.
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
    
        return (y - self.evolution(t, a, alpha, k, c, to, yo)) ** 2
    
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
        alpha = x[1]
        k = x[2]
        c = x[3]
        res = 0
        n = len(self.__data__)
        for t in range(n):
            res += self.error_sq(t, a, alpha, k, c, self.__to__,
                                 self.__data__[t], self.__yo__)
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
        return [self.evolution(t, a, alpha, k, c, self.__to__, self.__yo__)
                for t in range(limit)]

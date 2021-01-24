# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:18:21 2020

@author: lfiorentini
version 1.1
"""

from param_utilities import  process_param
from multiple_predictions import Multiple_predictions

class Param_dealer():
    def __init__(self, Cases, Deads, max_cases_p, max_deads_p, Population,
                 mort_thres_ratio, input_dir, output_dir, hour_format,
                 date_to_predict, top_contries):
        """
        Initializer
        Parameters : 
        
        Cases : pd.DataFrame
            Cases dataframe.
        Deads : pd.DataFrame
            Deads dataframe.
        max_cases_p : pd.DataFrame
            max cases dataframe.
        max_deads_p : pd.DataFrame
            max deads dataframe.
        Population : pd.DataFrame
            Population dataframe.
        mort_thres_ratio : float
            threshold for the mortality ratio between the current and the
            predicted one.
        input_dir : str
            str of the inout directory.
        output_dir : str
            str of the output directory.
        model : str
                str identifying the model used: possible values :
                    {'Verhulst', 'Gompertz', 'Mixed_VG', 'Allee'}.
        hour_format : str
            str for the hour format.
        date_to_predict : str
            str for the day to predict.
        top_contries : list
            list of the countries for the predictions.
    
        Returns :
        
        None.
    
        """
        self.__Cases__ = Cases
        self.__Deads__ = Deads
        self.__max_cases_p__ = max_cases_p
        self.__max_deads_p__ = max_deads_p
        self.__Population__ = Population
        self.__mort_thres_ratio__ = mort_thres_ratio
        self.__input_dir__ = input_dir
        self.__output_dir__ = output_dir
        self.__date_to_predict__ = date_to_predict
        self.__hour_format__ = hour_format
        self.__top_contries__ = top_contries
    
    def deal_w_param(self, Cases_p, Deads_p, model):
        """
        add the scale of the data, make predictions and process parameters

        Parameters :
        
        Cases_p : pd.DataFrame
            Cases parameters dataframe.
        Deads_p : pd.DataFrame
            Deads parameters dataframe.
        model : str
            model name.

        Returns :
        
        Cases_p : pd.DataFrame
            Cases parameters dataframe.
        Deads_p : pd.DataFrame
            Deads parameters dataframe.
        param_T : pd.DataFrame
            Parameters dataframe.
        unreal_ls : list
            list of countries with unrealistic predictons.

        """
        "add the scale of the data"
        Cases_p = Cases_p.merge(self.__max_cases_p__, on = 'country',
                                        how ='inner')
        Deads_p = Deads_p.merge(self.__max_deads_p__, on = 'country',
                                        how ='inner')
        MPV = Multiple_predictions(self.__Cases__, self.__Deads__, Cases_p,
                                   Deads_p, self.__input_dir__, 
                                   self.__output_dir__, model,
                                   self.__hour_format__,
                                   self.__date_to_predict__,
                                   self.__top_contries__)
        MPV.predict()
        param_T, unreal_ls = process_param(self.__Cases__, self.__Deads__,
                                           Cases_p, Deads_p,
                                           self.__Population__,
                                           self.__mort_thres_ratio__,
                                           self.__input_dir__, 
                                           self.__output_dir__, model)
        return Cases_p, Deads_p, param_T, unreal_ls
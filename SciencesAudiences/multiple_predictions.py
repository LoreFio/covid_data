# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:11:26 2020

@author: lfiorentini
version 1.1
"""

from os.path import join as path_join
from covid_predictor import Covid_predictor
from covid_utilities import stop_day

class Multiple_predictions:
    def __init__(self, Cases, Deads, param_cases, param_deads, input_dir,
                 output_dir, model, hour_format, date_to_predict,
                 top_contries):
        """
        Initializer
        Parameters : 
        
        Cases : pd.DataFrame
            Cases dataframe.
        Deads : pd.DataFrame
            Deads dataframe.
        param_cases : pd.DataFrame
            Cases parameters dataframe.
        param_deads : pd.DataFrame
            Deads parameters dataframe.
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
        self.__param_cases__ = param_cases
        self.__param_deads__ = param_deads
        self.__input_dir__ = input_dir
        self.__output_dir__ = output_dir
        self.__model__ = model
        self.__date_to_predict__ = date_to_predict
        self.__hour_format__ = hour_format
        self.__top_contries__ = top_contries
        self.__stop_day__ = stop_day
    
    def predict(self):
        """
        Make predictions

        Returns
        
        None.

        """
        k1Cases = list(self.__param_cases__[self.__param_cases__['k'] == 1][
            'country'])
        k1Deads = list(self.__param_deads__[self.__param_deads__['k'] == 1][
            'country'])
        
        for country in self.__top_contries__ :
            outputfile = path_join(self.__output_dir__, self.__model__,
                                   country + '.txt')
            with open(outputfile, 'w') as f:
                print("\n", country, file = f)
                if country in k1Cases:
                    print('It seems that it has reached the maximum for cases',
                          file = f)
                if country in k1Deads:
                    print('It seems that it has reached the maximum for deads',
                          file = f)
                Ca_pred = Covid_predictor(country, self.__param_cases__,
                                          self.__input_dir__,
                                          path_join(self.__output_dir__,
                                                    self.__model__),
                                          self.__Cases__,
                                          model = self.__model__,
                                          label = 'Cases')
                De_pred = Covid_predictor(country, self.__param_deads__,
                                          self.__input_dir__,
                                          path_join(self.__output_dir__,
                                                    self.__model__),
                                          self.__Deads__,
                                          model = self.__model__,
                                          label = 'Deads')
                
                Ca_day = Ca_pred.predict_until(self.__date_to_predict__)
                De_day = De_pred.predict_until(self.__date_to_predict__)
            
                Ca_day = Ca_day[-1] * Ca_pred.__total__
                De_day = De_day[-1] * De_pred.__total__
                
                print("current Cases:", Ca_pred.current_total(),
                      file = f)
                print("prediction Cases for ", self.__date_to_predict__, ":", 
                      Ca_day, file = f)
                if self.__stop_day__ == self.__date_to_predict__:
                    print("real Cases for ", self.__date_to_predict__, ":",
                          Ca_pred.stop_day_value(), file = f)

                print("prediction total Cases:", Ca_pred.predict_total(),
                      file = f)
                print("error margin Cases:", Ca_pred.error_margin(), file = f)
                
                print("current Deads:", De_pred.current_total(),
                      file = f)
                print("prediction Deads for ", self.__date_to_predict__, ":",
                      De_day, file = f)
                if self.__stop_day__ == self.__date_to_predict__:
                    print("real Deads for ", self.__date_to_predict__, ":",
                          De_pred.stop_day_value(), file = f)

                print("prediction total Deads:", De_pred.predict_total(),
                      file = f)
                print("error margin Deads:" ,De_pred.error_margin(), file = f)
                f.close()
                
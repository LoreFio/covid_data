# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:10:46 2020

@author: lfiorentini
"""
import pandas as pd
from os.path import join as path_join
from datetime import datetime
from covid_utilities import plot_prediction, hour_format, initial_day, stop_day
from verhulst import Verhulst
from gompertz import Gompertz
from mixed_VG import Mixed_VG
from allee import Allee
from avg import Avg

class Covid_predictor():
    def __init__(self, country, param, input_dir, output_dir, data,
                 model = 'Verhulst', label = 'Cases'):
        """
        Constructor of the class

        Parameters :
        
        country : str
            country name.
        param : pd.DataFrame
            DataFrame with the parameters.
        input_dir : str
            input directory.
        output_dir : str
            output directory.
        data : pd.DataFrame
            DataFrame with the historical data.
        data_der : pd.DataFrame
            DataFrame with the historical data derivative. The default is
            'None'.
        model : str, optional
            str to identify the model used: possible values :
                {'Verhulst', 'Gompertz', 'Mixed_VG', 'Allee'}.
                The default is 'Verhulst'.
        label : str, optional
            str in {'Cases', 'Deads'}. The default is 'Cases'.

        Returns :
        
        None.

        """
        self.__country__ = country
        line = param[param['country'] == country]

        self.__line__ = line
        self.__total__ = int(line['total_n'])
        self.__err__ = float(line['err_real_data'])
        self.__output_dir__ = output_dir 
        self.__data__ = data
        if model == 'Verhulst':
            self.__mod__ = Verhulst(data[self.__country__])
        elif model == 'Gompertz':
            self.__mod__ = Gompertz(data[self.__country__])    
        elif model == 'Mixed_VG':
            self.__mod__ = Mixed_VG(data[self.__country__])
        elif model == 'Allee':
            self.__mod__ = Allee(data[self.__country__], None)
        elif model == 'AVG':
            self.__mod__ = Avg(data[self.__country__], None)
        self.__label__ = label
        
        self.__stop_day__ = stop_day
        if label == 'Cases':
            self.__stop_day__datum = pd.read_csv(path_join(input_dir,
                                                         'stop_day_cases.csv'),
                                               sep = ';')[self.__country__]
        elif label == 'Deads':
            self.__stop_day__datum = pd.read_csv(path_join(input_dir,
                                                         'stop_day_deads.csv'),
                                               sep = ';')[self.__country__]

    def stop_day_value(self):
        """
        This method returns the real datum for stop_day

        Returns : 
        
        int64
            Real datum for stop_day.

        """
        return self.__stop_day__datum

    def stop_day_rel_err(self):
        """
        This method returns the real datum for stop_day

        Returns : 
        
        int64
            Real datum for stop_day.

        """
        pred = self.predict_until(self.__stop_day__, 'stop_day_pred',
                                  do_plt = False)
        scaled_pred = self.__total__ * float(pred[-1])
        return abs(
            self.__stop_day__datum - scaled_pred) / self.__stop_day__datum
    
    def predict_until_n(self, prediction_days, date_to_predict,
                        name = 'pred', do_plt = True):
        """
        Predict the future outcome until the given date using historical data

        Parameters :
        
        prediction_days : int
            int of the day to predict.
        date_to_predict : str
            str with the ending day of the prevision.
        name : str
            str with the name for the graph. The default is 'pred'.
        do_plt : bool
            if True the plot will be done. The default is True
        Returns :
        
        pred: list
           scaled result of the prediction from the first day to 
           date_to_predict 
        
        """
            
        pred = self.__mod__.prediction_series(self.__line__,
                                              limit = prediction_days)  
        if do_plt:
            plt_str = self.__country__ + "_" + self.__label__ + name
            plot_prediction(self.__data__['date'],
                            self.__data__[self.__country__], pred,
                            self.__output_dir__, err = self.__err__,
                            index_2 = range(len(pred)), name = plt_str)
        return pred
    
    def predict_until(self, date_to_predict, name = 'pred', do_plt = True):
        """
        Predict the future outcome until the given date using historical data

        Parameters :
        
        date_to_predict : str
            str with the ending day of the prevision.
        name : str
            str with the name for the graph. The default is 'pred'.
        do_plt : bool
            if True the plot will be done. The default is True

        Returns :
        
        pred: list
           scaled result of the prediction from the first day to 
           date_to_predict 

        """
        prediction_date = datetime.strptime(date_to_predict,
                                            hour_format)
        prediction_days = (prediction_date - initial_day).days
        return self.predict_until_n(prediction_days, date_to_predict, name,
                                    do_plt)
    
    def current_total(self):
        """
        This method returns the current total

        Returns :
        
        int
            current total.

        """
        return self.__total__

    def predict_total(self):
        """
        This method returns the predicted total

        Returns :
        
        float
            predicted total.

        """

        return self.__total__ * self.__mod__.predict_max(self.__line__)
    
    def error_margin(self):
        """
        This method returns the predicted total

        Returns :
        
        float
            predicted total.

        """

        return self.__total__ * self.__err__
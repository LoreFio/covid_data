# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:00:03 2020

@author: lfiorentini
version 1.1
"""

from covid_predictor import Covid_predictor

class Model_error():
    def __init__(self, param, input_dir, output_dir, data, model = 'Verhulst',
                 label = 'Cases'):
        """
        Initialiser

        Parameters : 
        
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
        self.__param__ = param
        self.__input_dir__ = input_dir 
        self.__output_dir__ = output_dir
        self.__data__ = data
        self.__model__ = model
        self.__label__ = label

    def stop_day_rel_err(self, country):
        """
        Compute the today_rel_err for a country

        Parameters : 
        
        country : str
            country name.

        Returns : 
        
        None.

        """
        try:
            predictor = Covid_predictor(country, self.__param__,
                                        self.__input_dir__, self.__output_dir__,
                                        self.__data__, self.__model__,
                                        self.__label__)
            return predictor.stop_day_rel_err()
        except ValueError:
            print(country)
            return 1
        except KeyError:
            print(country)
            return 1
        
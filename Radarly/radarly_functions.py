# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:24:06 2020

@author: lfiorentini
"""
import re
import nltk

def lst_contains(lst, word):
    for el in lst:
        if el == word:
            return True
    return False

def remove_columns(Data, col_elem):
    """
    This function remove inplace all the columns containing a given string

    Parameters : 
    
    Data : pd.DataFrame
        Dataframe to filter.
    col_elem : str
        str that identifies columns to be removed.


    """
    Data.drop([el for el in Data.columns if col_elem in el], axis = 1,
              inplace = True)

def spacy_process(message, sub_dict, stop_list, spacy_nlp):
    """
    Process a text using spacy nlp

    Parameters :
    
    message : str
        text to be processed.
    sub_dict : dict
        dictionary of regex to apply.
    stop_list : iterable
        list of stopwords.
    spacy_nlp : spacy.lang
        spacy.lang used for nlp preprocessing
    Returns :
    
    list
        tokenized and processed text without stopwords.

    """
    for key in sub_dict:
        message = re.sub(key, sub_dict[key], message)
    return [tok.lemma_ for tok in spacy_nlp(message.lower())
            if tok.lemma_ not in stop_list and tok not in stop_list]
    
def remove_stopwords(message, stop_list, processer):
    """
    Remove stopwords from a tokenized message

    Parameters :
    
    message : list
        list of tokens.
    stop_list : list
        list of stopwords.

    Returns :
    
    list
        list of tokens.

    """
    return list(set(message) - set(stop_list))

def exctract_freq(Data, col, sep, replacer = None):
    """
    This method take a column inside the dataset, group its element divided by
    the separator indicated and return a table with the distribution of the
    elements.

    Parameters :
    
    Data : pd.DataFrame
        Dataframe to be used.
    col : str
        Name of the column to be processed.
    sep : str
        separator inside the string.
    replacer: tuple
        tuple containing the element to replace and the replacer.
        The default value is None

    Returns :
    
    nltk.probability.FreqDist
        Distribution of the element inside the column.

    """
    lst = []
    for el in Data[col].fillna(""):
        if len(el) > 0:
            t_lst = str(el).split(sep = sep)
            if replacer != None:
                t_lst = [el.replace(replacer[0], replacer[1]) for el in t_lst]
            lst += t_lst
        
    return nltk.FreqDist(lst)


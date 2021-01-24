# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:41:18 2020

@author: lfiorentini
"""

import re
import pandas as pd
import numpy as np
import time
import spacy
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/lfiorentini/TequilaRapido/scraper/code')
# class to tokenize strings with the nltk built-in word tokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from radarly_functions import spacy_process, exctract_freq, remove_columns
    
start_time = time.time()

'''
1 file for january, february and march, 3 files for april divided as 1->6 6->20
21->30 
'''

preprocess_tool = "spacy"

Data = pd.concat([pd.read_csv('covid_radarly_mars.csv', sep = ';',
                              low_memory = False),
                  pd.read_csv('covid_radarly_avri.csv', sep = ';',
                              low_memory=False),
                  pd.read_csv('covid_radarly_avri_2.csv', sep = ';',
                              low_memory=False),
                  pd.read_csv('covid_radarly_avri_3.csv', sep = ';',
                              low_memory=False)], sort = True)
"""
Data = pd.read_csv('covid_radarly_avri_3.csv', sep = ';', low_memory = False)
"""
Data.reset_index()
Data.drop_duplicates(subset = None, keep = 'first', inplace = True)

one_val_col = [el for el in Data.columns if len(Data[el].unique()) == 1 ]

Data.drop(one_val_col, axis = 1, inplace = True)

remove_columns(Data, 'declared')
remove_columns(Data, 'id')
remove_columns(Data, 'last update of the')

Data.drop(['permalink', 'avatar', 'embedded url', 'screen name', 'media url',
           'story'], axis = 1, inplace = True)

"""
From now on we remove columns that could be useful in further studies

"""

remove_columns(Data, 'image')
remove_columns(Data, 'inferred')
remove_columns(Data, 'facebook')
remove_columns(Data, 'instagram')
remove_columns(Data, 'twitter')
remove_columns(Data, 'web')

Data.drop(['verified author account', 'reviews Radarly normalized rating',
           'reviews platform rating'], axis = 1, inplace = True)
emotion_set = [el for el in set(Data['emotion(s)']) if ';' not in str(el)]     
Data = Data[Data['text'].notna()]
Data.drop_duplicates(keep='first', inplace = True)

load_time = time.time() 
print("Loading and removing useless columns time: %s seconds "
      % (load_time - start_time))

"""
NLP

"""

hashtags = exctract_freq(Data, 'hashtags', ';')
mentions = exctract_freq(Data, 'mentions', ';')
entities = exctract_freq(Data, 'named entities', ';', replacer = ('.', ' '))
Data.drop(['named entities'], axis = 1, inplace = True)

topnum = 100
top_hashtags = hashtags.most_common(topnum)
top_mentions = mentions.most_common(topnum)
top_entities = entities.most_common(topnum)

positif_words = ['applaudit', 'soignants', 'soigner', 'gel', 'hydroalcoolique',
                 'solidarité', 'solidarite']
incite_words = ['rester', 'restez', 'reste', 'restons']
negative_words = ['morts', 'epidemie', 'épidémie', 'crise']

language = "french"
normalizer = "SnowballStemmer"

sub_dict = {r"\+": " plus ",
            r"\#": " ",
            r"\@": "",
            #r'[Cc][Oo][Vv][Ii][Dd][0-9]*[-‐_ .?ー—–]*[0-9]*[fr]*': "covid ",
            #r'(?i)covid[0-9]*[-‐_ .?ー—–\/]*[0-9]*[france]*': "covid ",
            r'(?i)covid[0-9]*[^\w\s]*[0-9]*[france]*': "covid ",
            # unify covid19
            r'(?i)corona[-_ー ]*virus': "covid ",
            # unify coronavirus
            r'(?i)C.O.V.I.D-19': "covid ",
            # unify C.O.V.I.D-19
            r'(?i)corona[-]*[0-9]*': "covid ",
            # unify corona
            r'ー19': "covid ",
            # unify corona
            r'(?i)confinementjour[0-9]+': "confinement jour",
            # unify pandémie
            r'(?i)pandemie': "pandémie",
            # unify épidémie
            r'(?i)epidemie': "épidémie",
            # unify confinement
            r'[0-9]+[eè][rm][e]*': "",
            # remove days of confinement
            r'(?i)plusjamais': "plusjamais ",
            # break plusjamaisqqch texts
            r'[0-9]+h[0-9]*': "",               # hours
            r'\xa0': " ",                       # white space
            r'\u2009': " ",                     # white space
            r'\n': "",                          # remove \n
            r'\r': "",                          # remove \r
            r'\t': "",                          # remove \t
            r'&amp': "",                        # remove &
            r'[\"\'\u2018\u2019]': " ",         # remove quotes
            r'www.[^ ]*': "",                   # remove links
            r'http[s]*[\/]*[^ ]*': "",          # remove links
            #r'[^ ]*@[^ ]*': "",                # remove mail addresses
            r'[^ ]*_[^ ]*': "",                 # remove words with _
            r'([\U00002600-\U000027BF])|\
                ([\U00010000-\U0010ffff])': "", # emojis and others
            r'[^\w\s\'\’]': " ",                # punctuations
            r' +': " ",                         # multiple spaces
            r'^\s': "",                         # spaces at the beginning
            }

fr_stop = stopwords.words('french') + ['celleci', 'celui', 'celuici', 'cet',
                                       'ceux', 'chaqun']

spacy_nlp = spacy.load("fr_core_news_md", disable=["parser", "ner"])
stop_list = set(tok.lemma_ for tok in spacy_nlp(' '.join(fr_stop)))
stop_list.update(spacy_nlp.Defaults.stop_words)

Data['processed'] = Data['text'].apply(spacy_process,
                                       sub_dict = sub_dict,
                                       stop_list = stop_list,
                                       spacy_nlp = spacy_nlp)  

Data.drop('text', axis = 1, inplace = True)
nlp_clean_time = time.time() 
print("NLP cleaning time: %s seconds "
      % (nlp_clean_time - load_time))

w2v_model = Word2Vec(min_count = 50, size = 100, window = 4,
                     workers = multiprocessing.cpu_count()-1)
w2v_model.build_vocab(Data['processed'], progress_per = 10000)
w2v_model.train(Data['processed'], total_examples = w2v_model.corpus_count,
                epochs=30, report_delay=1)

word_vectors = w2v_model.wv
print("corpus size:", len(word_vectors.vocab.keys()))
print("words similar to covid:",
      word_vectors.most_similar(positive = ['covid'], topn = 10))
print("words similar to crise:",
      word_vectors.most_similar(positive = ['cris'], topn = 10))
print("words similar to confinement:",
      word_vectors.most_similar(positive = ['confinement'], topn = 10))
print("words similar to italie:",
      word_vectors.most_similar(positive = ['italie'], topn = 10))
w2v_time = time.time() 
print("w2v time: %s seconds "
      % (w2v_time - nlp_clean_time ))

"""
clustering 2 clusters

"""
KM_model2 = KMeans(n_clusters = 2, max_iter = 1000, random_state = True,
                   n_init = 50).fit(X = word_vectors.vectors)
positive_cluster_center = KM_model2.cluster_centers_[0]
negative_cluster_center = KM_model2.cluster_centers_[1]

g0 = word_vectors.similar_by_vector(KM_model2.cluster_centers_[0], topn = 15,
                               restrict_vocab = None)
g1 = word_vectors.similar_by_vector(KM_model2.cluster_centers_[1], topn = 15,
                               restrict_vocab = None)
g0_l = str([el[0] for el in g0])
g1_l = str([el[0] for el in g1])

score0_covid = [word_vectors.similarity('covid', el[0]) for el in g0]
score1_covid = [word_vectors.similarity('covid', el[0]) for el in g1]
score0_confinement = [word_vectors.similarity('confinement', el[0])
                      for el in g0]
score1_confinement = [word_vectors.similarity('confinement', el[0])
                      for el in g1]

"""
g2 = word_vectors.similar_by_vector(KM_model2.cluster_centers_[2], topn = 15,
                               restrict_vocab = None)
"""
metric_str = 'euclidean'
score = silhouette_score(word_vectors.vectors,
                         KM_model2.predict(word_vectors.vectors),
                         metric = metric_str)
print("silhouette_score:", score)

SVmodel = SilhouetteVisualizer(KM_model2, is_fitted = True)
SVmodel.fit(word_vectors.vectors)
SVmodel.show()  

words = pd.DataFrame(word_vectors.vocab.keys(), columns = ['words'])
words['vectors'] = words.words.apply(lambda x: word_vectors[f'{x}'])
words['cluster'] = words.vectors.apply(lambda x: KM_model2.predict(
    [np.array(x)]))
words.cluster = words.cluster.apply(lambda x: x[0])
words['cluster_value'] = [1 if i == 0 else -1 for i in words.cluster]
words['closeness_score'] = words.apply(
    lambda x: 1/(KM_model2.transform([x.vectors]).min()), axis = 1)
words['sentiment_coeff'] = words.closeness_score * words.cluster_value

clus_time = time.time() 
print("clustering time: %s seconds "
      % (clus_time - w2v_time))  

"""
clustering 3 clusters

"""

KM_model3 = KMeans(n_clusters = 3, max_iter = 1000, random_state = True,
                  n_init = 50).fit(X = word_vectors.vectors)
positive_cluster_center = KM_model3.cluster_centers_[0]
negative_cluster_center = KM_model3.cluster_centers_[1]

g03 = word_vectors.similar_by_vector(KM_model3.cluster_centers_[0], topn = 15,
                               restrict_vocab = None)
g13 = word_vectors.similar_by_vector(KM_model3.cluster_centers_[1], topn = 15,
                               restrict_vocab = None)
g23 = word_vectors.similar_by_vector(KM_model3.cluster_centers_[2], topn = 15,
                               restrict_vocab = None)
g03_l = str([el[0] for el in g03])
g13_l = str([el[0] for el in g13])
g23_l = str([el[0] for el in g23])

score03_covid = [word_vectors.similarity('covid', el[0]) for el in g03]
score13_covid = [word_vectors.similarity('covid', el[0]) for el in g13]
score23_covid = [word_vectors.similarity('covid', el[0]) for el in g23]
score03_confinement = [word_vectors.similarity('confinement', el[0])
                       for el in g03]
score13_confinement = [word_vectors.similarity('confinement', el[0])
                       for el in g13]
score23_confinement = [word_vectors.similarity('confinement', el[0])
                       for el in g23]

metric_str = 'euclidean'
score = silhouette_score(word_vectors.vectors,
                         KM_model3.predict(word_vectors.vectors),
                         metric = metric_str)
print("silhouette_score:", score)

SVmodel3 = SilhouetteVisualizer(KM_model3, is_fitted = True)
SVmodel3.fit(word_vectors.vectors)
SVmodel3.show()  

words['cluster_3'] = words.vectors.apply(lambda x: KM_model3.predict(
    [np.array(x)])[0])
words.drop(['vectors', 'cluster_value'], axis = 1, inplace = True)
clus_time2 = time.time() 
print("clustering time: %s seconds "
      % (clus_time2 - clus_time))  

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:47:10 2020

@author: lfiorentini
"""

import pandas as pd
from os.path import join as path_join
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE

output_dir = './output'
model = 'Verhulst'
#model = 'Gompertz'
param = pd.read_csv(path_join(output_dir, model, 'param.csv'), sep = ';',)
Q1 = param.drop(['country', 'continent'], axis = 1).corr()

scaled = param.drop(['country', 'continent', 'n_data_cases',
                           'n_data_deads', 'total_n_cases', 'total_n_deads',
                           'err_real_data_cases', 'err_real_data_deads',
                           'predict_total_cases', 'predict_total_deads',
                           'estimated_completion_cases',
                           'estimated_completion_deads'], axis = 1)
scaler = MinMaxScaler().fit(scaled)
scaled = pd.DataFrame(scaler.transform(scaled),
                            columns = scaled.columns)

scaled.to_csv(path_join(output_dir, model, 'scaled.csv'),
                      index = False)
precision = 0.90

pca_obj = PCA(precision)
pc_data = pca_obj.fit_transform(scaled)
print(pca_obj.explained_variance_ratio_)

pca_load = pca_obj.components_
print(pca_load.shape)
pca_load_3D = pd.DataFrame(pca_load[0:3, :], columns = scaled.columns)

PCA_Frame = pd.DataFrame(pc_data[:, 0:3], columns = ['PC1', 'PC2', 'PC3'])
PCA_Frame['country'] = param['country']

fig, ax = plt.subplots()
ax.scatter(PCA_Frame['PC1'], PCA_Frame['PC2'])

if model == 'Verhulst':
    labx = 'high mortality'
    laby = 'slow evolution and high totals'
elif model == 'Gompertz':
    labx = 'slow evolution and high totals'
    laby = 'high mortality and fast cases'
    
plt.xlabel(labx)
plt.ylabel(laby)

for i in range(len(PCA_Frame)):
    if PCA_Frame['country'][i] in ['China', 'United States']:
        cur_c = 'red'
    else:
        cur_c = 'blue'
    ax.annotate(PCA_Frame['country'][i], (PCA_Frame['PC1'][i],
                                          PCA_Frame['PC2'][i]), color = cur_c)

plt.title('Distribution of a in countries')
plt.savefig(path_join(output_dir, model, 'PCA_.png'))

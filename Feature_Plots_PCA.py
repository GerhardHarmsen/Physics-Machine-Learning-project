# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:29 2020

@author: gerha
"""

import numpy as np
from random import randint
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                                and (abs(i[1]-x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height:
                differ = np.diff(sorted_ltp, axis = 0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    if j > txt_height * 1.5:
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text_positions, labels, axis,txt_width,txt_height):
    i = 0
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - .03, 1.02*t, labels[i],rotation=0, color='black', fontsize=13)
        i = i + 1
        if y != t:
            axis.arrow(x, t,0,y-t, color='black',alpha=0.2, width=txt_width*0.1,
                       head_width=.02, head_length=txt_height*0.5,
                       zorder=0,length_includes_head=True)

def PLTColumns(column1, column2):
  XData = []
  YData = []
  for i in range(dtrain.shape[0]):
      if dtrain[i,column1] != -999 and dtrain[i,column2] != -999:
         XData.append(dtrain[i,column1]) 
         YData.append(dtrain[i,column2])
  fig, ax = plt.subplots()
  fig = plt.scatter(XData,YData);
  ax.set_xlabel(xlabel = columnNames[column1])
  ax.set_ylabel(ylabel = columnNames[column2])
  plt.show()

def FeaturePlots(DataSet):
    sns.pairplot(DataSet.loc[:,DataSet.dtypes == 'float64'])
    corr = DataSet.loc[:, DataSet.dtypes == 'float64'].corr()
    plt.title('Linear correlation plot of the features in the dataset')
    plt.show()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
    plt.title('Heat map of the features showing linear correlation of the features')
    plt.show()

def PCAAnalysis(DataSet, LabelOfInterest):
    DataSet2 = DataSet.drop(labels = LabelOfInterest, axis = 1)
    print(DataSet2.describe())
    for col in np.unique(DataSet.columns):
        plt.figure(num =None, figsize = [20, 20])
        boxplot1 = DataSet.boxplot(by = LabelOfInterest, column = col)
        plt.show()
    
    scalar = StandardScaler()
    scalar.fit(DataSet2)
    scaled_data =scalar.transform(DataSet2)
    pca = PCA(n_components = 2)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    plt.figure(num =None, figsize = [20, 20])   
    for g in tqdm(np.unique(DataSet[LabelOfInterest])):
        i = np.where(np.abs(DataSet[LabelOfInterest]) == g)
        plt.scatter(x_pca[i,0], x_pca[i,1], c = '#%06X' % randint(0, 0xFFFFFF), label = g )
    plt.title('Principal Component plot of the data')
    plt.legend()
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()      
    updatedColumns = list(DataSet2.columns)
    df_comp= pd.DataFrame(pca.components_,columns = updatedColumns)
    fig, ax = plt.subplots(figsize = (20, 20))
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel('')
    ax.scatter(pca.components_[0,:],pca.components_[1,:])
    Circ = plt.Circle([0,0], radius = 1, fill = None)
    ax.add_patch(Circ)
    plt.title('Weighting of the features in the dataset for the first and second prinipal componets')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    ax.grid()
    txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.04*(plt.xlim()[1]- plt.xlim()[0])
    text_positions = get_text_positions(pca.components_[0,:],pca.components_[1,:],txt_width,txt_height)
    text_plotter(pca.components_[0,:], pca.components_[1,:], text_positions,updatedColumns , ax , txt_width,txt_height)
    
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
import click

TitleSize = 20
FontSize  = 15
plt.rc('legend',fontsize=20)

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
        axis.text(x - .03, 1.02*t, labels[i],rotation=0, color='black', fontsize=FontSize)
        i = i + 1
        if y != t:
            axis.arrow(x, t,0,y-t, color='black',alpha=0.2, width=0.014,
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
  ax.set_xlabel(xlabel = columnNames[column1], fontsize = FontSize)
  ax.set_ylabel(ylabel = columnNames[column2], fontsize = FontSize)
  plt.show()

def FeaturePlots(DataSet, LabelOfInterest):
    sns.pairplot(DataSet, hue = LabelOfInterest)
    plt.title('Linear correlation plot of the features in the dataset')
    plt.show()
    sns.heatmap(DataSet.corr(), xticklabels=DataSet.columns, yticklabels=DataSet.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = False)
    plt.title('Heat map of the features showing linear correlation of the features')
    plt.show()

def PCAPlots(DataSet,LabelOfInterest,NoofJets,NoofLepton,ax=None,ax1=None, plt_kwargs = {}, sct_kwargs = {}):
    DataSet2 = DataSet.drop(labels = LabelOfInterest, axis = 1)
    DataSet2.dropna(axis=1,inplace=True)
    scalar = StandardScaler()
    scalar.fit(DataSet2)
    scaled_data =scalar.transform(DataSet2)
    #### Scree plots
    #pca = PCA(n_components= len(DataSet2.columns))
    #pca.fit_transform(DataSet2)
    #plt.plot(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_)
    #plt.title('Scree plot', fontsize = TitleSize)
    #plt.xlabel('Number of PCA components', fontsize = FontSize)
    #plt.ylabel('Eigenvalue', fontsize = FontSize)
    #plt.show()
    ####
    #PCA_Num = click.prompt('From the screen plot how many PCA components would you like to use?', default = 2, type=int)
    pca = PCA(n_components = 2)

    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    if ax is None:
        ax = plt.gca()
   
    ColourCodes = { 0 : {'Lbl' : 'Background', 'Color' : 'Blue'},
                    1 : {'Lbl' : 'Signal',  'Color' : 'Orange'},
                    'True Negative' : {'Lbl' : 'True Negative', 'Color' : 'Blue'},
                    'False Negative' : {'Lbl' : 'False Negative', 'Color' : 'Orange'},
                    'False positive' : {'Lbl' : 'False positive', 'Color' : 'Green'},
                    'True positive' : {'Lbl' : 'True positive', 'Color' : 'Red'}     
                    }
    for g in np.unique(DataSet[LabelOfInterest]):
        i = np.where(DataSet[LabelOfInterest] == g)
        PercentageOfCase = (len(DataSet[DataSet[LabelOfInterest] == g]) / len(DataSet)) * 100
        ax.scatter(x_pca[i,0], x_pca[i,1], label = ColourCodes[g]['Lbl'] + ' | ' + str(np.round(PercentageOfCase)) + '%', c = ColourCodes[g]['Color'])
    #ax.set_title('Principal Component plot of the data for {} jets and {} leptons'.format(NoofJets,NoofLepton))    
    
    updatedColumns = list(DataSet2.columns)
    df_comp= pd.DataFrame(pca.components_,columns = updatedColumns)
    if ax1 is None:
        ax1 = plt.gca()
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_xlabel('')
    ax1.scatter(pca.components_[0,:],pca.components_[1,:])
    Circ = plt.Circle([0,0], radius = 1, fill = None)
    ax1.add_patch(Circ)
    ax1.grid()
    txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.2*(plt.xlim()[1]- plt.xlim()[0])
    text_positions = get_text_positions(pca.components_[0,:],pca.components_[1,:],txt_width,txt_height)
    text_plotter(pca.components_[0,:], pca.components_[1,:], text_positions,updatedColumns , ax1 , txt_width,txt_height)
    return ax, ax1

def PCAAnalysis(DataSet, LabelOfInterest,BoxPlot = False):
    if BoxPlot:
        DataSet2 = DataSet.drop(labels = LabelOfInterest, axis = 1)
        for col in np.unique(DataSet2.columns):
            plt.figure(num =None, figsize = [20, 20])
            boxplot1 = DataSet.boxplot(by = LabelOfInterest, column = col)
            plt.show()
    
    
    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=(40, 40))
    fig1, axes1 = plt.subplots(nrows = 3, ncols = 3, figsize=(40, 40))
    
    for Jets in [0,1,2]:
        for Leptons in [0,1,2]:
            if Jets == 2:
                PCADataSet = DataSet[DataSet.PRI_jets >= 2]
            else:
                PCADataSet = DataSet[DataSet.PRI_jets == Jets]
            
            if Leptons == 2:
                PCADataSet = PCADataSet[PCADataSet.PRI_nleps >= 2] 
            else:
                PCADataSet = PCADataSet[PCADataSet.PRI_nleps == Leptons] 
            
            try:
                PCAPlots(PCADataSet,LabelOfInterest,Jets,Leptons, ax=axes[Jets,Leptons],ax1=axes1[Jets,Leptons])      
            except:
                pass
            
            axes[Jets,Leptons].legend()
            axes[2,Leptons].set_xlabel('Number of leptons: {}'.format(Leptons),FontSize = 25)
            axes[Jets,0].set_ylabel('Number of Jets: {}'.format(Jets), FontSize = 25)
            axes1[2,Leptons].set_xlabel('Number of leptons: {}'.format(Leptons),FontSize = 25)
            axes1[Jets,0].set_ylabel('Number of Jets: {}'.format(Jets), FontSize = 25)
            
    
    plt.tight_layout() #This to avoid overlap of labels and titles across plots
    plt.show()
        
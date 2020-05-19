# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:26:58 2020

@author: gerha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def DictionaryPlot(DictList, ChartName):
  plt.figure(figsize = (20, 20))
  plt.bar(range(len(DictList)),DictList.values()) 
  plt.title(ChartName)
  plt.ylabel('Relevance')
  plt.xticks(ticks = range(len(DictList)), labels = list(DictList.keys()), rotation=90)
  plt.show()

def XGBoostersConfusionMatrix(DataSet, Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    print("Training model")
    model = XGBClassifier()
    if len(np.unique(Y)) == 2:
        model.fit(X = X_train, y = y_train, eval_metric = "error" ,eval_set = [(X_test, y_test)], verbose = True, early_stopping_rounds = 10)
    else:   
        model.fit(X = X_train, y = y_train, eval_metric = "merror" ,eval_set = [(X_test, y_test)], verbose = True, early_stopping_rounds = 10)   
    print(model)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    GXBoost_confusion = plot_confusion_matrix(model, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize = 'true')

    GXBoost_coeff = dict(zip(DataSet.columns,
                        np.round(np.concatenate((model.feature_importances_), axis=None), 3)))
    print('GXBoost coefficients:{}'.format(GXBoost_coeff))
    plt.figure(figsize = (20, 20))
    plt.bar(range(len(GXBoost_coeff)),GXBoost_coeff.values()) 
    plt.title("GXBoost")
    plt.ylabel('Relevance')
    plt.xticks(ticks = range(len(GXBoost_coeff)), labels = list(GXBoost_coeff.keys()), rotation=90)
    plt.show()


def XGBoostersFeatureComparison(DataSet, Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    results = pd.DataFrame(columns=['num_features','features','Accuracy', 'ConfusionMatrix'])
    print("Training models")
    model = XGBClassifier()
    if len(np.unique(Y)) == 2:
      for k in range(1, X_train.shape[1] + 1):
        for subset in tqdm(itertools.combinations(range(X_train.shape[1]), k), leave = None):
            subset = list(subset)
            model.fit(X = X_train[X_train.columns[subset]], y = y_train, eval_metric = "error" ,eval_set = [(X_test[X_test.columns[subset]], y_test)], verbose = False, early_stopping_rounds = 10)
            y_pred = model.predict(X_test[X_test.columns[subset]])
            predictions = [value for value in y_pred]
            accuracy = accuracy_score(y_test, predictions)
            XGBoost_confusion = confusion_matrix(y_test, y_pred, normalize = 'true')
            results = results.append(pd.DataFrame([{'num_features' : k,
                                                  'features' : DataSet.columns[subset],
                                                  'coeffs' : np.round(model.feature_importances_),
                                                  'Accuracy' : accuracy,
                                                  'ConfusionMatrix' : XGBoost_confusion}]))   
            
    else: 
        for k in range(1, X_train.shape[1] + 1):
            for subset in tqdm(itertools.combinations(range(X_train.shape[1]), k), leave = None):
                subset = list(subset)  
                model.fit(X = X_train, y = y_train, eval_metric = "merror" ,eval_set = [(X_test, y_test)], verbose = True, early_stopping_rounds = 10)   
                y_pred = model.predict(X_test[X_test.columns[subset]])
                predictions = [round(value) for value in y_pred]
                accuracy = accuracy_score(y_test, predictions)
                XGBoost_confusion = confusion_matrix(y_test, y_pred, normalize = 'true')
                results = results.append(pd.DataFrame([{'num_features' : k,
                                                  'features' : DataSet.columns[subset],
                                                  'coeffs' : np.round(np.concatenate((model.feature_importances_))),
                                                  'Accuracy' : accuracy,
                                                  'ConfusionMatrix' : XGBoost_confusion}]))
    results = results.sort_values('Accuracy').reset_index()
    BestScore = results['Accuracy'].max()
    Features = {i : 0 for i in DataSet.columns}
    for x in range(len(results)):
        if results['Accuracy'][x] == BestScore:
            sns.heatmap(results['ConfusionMatrix'][x], annot =True, cmap=plt.cm.Blues,)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(dict(zip(results['features'][x],results['coeffs'][x])))
            plt.show()
            #DictionaryPlot(dict(zip(results['features'][x].tolist(),results['coeffs'][x].tolist())), 'Logistic Linear Regression with accuracy {}'.format(results['Accuracy'][x]))
            #print('Features for top result :{}'.format(dict(zip(results['features'][x].tolist(),results['coeffs'][x].tolist()))))
            for i in range(len(results['features'][x])):
                if results['coeffs'][x][i] != 0:
                   Features[results['features'][x][i]] = Features[results['features'][x][i]] + 1
                
    Features =Features = {k: v for k, v in sorted(Features.items(), key = lambda item: item[1], reverse = True)}  
    print(Features)  
    DictionaryPlot(Features, "Frequency of features in best models")
    return results 
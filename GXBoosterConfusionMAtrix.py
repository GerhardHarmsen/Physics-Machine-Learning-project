# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:26:58 2020

@author: gerha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from numpy import loadtxt
import xgboost as xgb
from xgboost import plot_tree
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import Feature_Plots_PCA

def HyperParameterTuning(DataSet, Y):
    """
    Function for selecting the optimum values for the hyperparmeters to use for the XGBoost model.

    Parameters
    ----------
    DataSet : Pandas dataset
        Dataset containing the features to be used for training the model. There should be no weights included in this dataset.
    Y : One dimensional array
        Array or dataset containing the labels relating to the training dataset.

    Returns
    -------
    Dict.
        Returns a dictionary containing the optimum configuration of hyperparameters for creating the decision tree model.
        

    """
    test_size = 0.3
    seed = 0
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    param_grid =  { 'learning_rate' : [0.01, 0.1, 0.5, 0.9],
                   'n_estimators' : [10, 50, 10, 150, 200],
                   'subsample': [0.3, 0.5, 0.9],
                   'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'base_score' : [0.1, 0.5, 0.9],
                   'reg_alpha' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                   'min_split_loss' : [0, 0.5, 0.8, 1, 2],
                   'reg_gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ]}
    
    model = XGBClassifier()
    randomized_mse = RandomizedSearchCV(estimator = model, param_distributions=param_grid,n_iter = 200, scoring='roc_auc', n_jobs =-1, cv=4, verbose = 1)
    randomized_mse.fit(X_train, y_train)
    print('Best parameters found: ', randomized_mse.best_params_)
    print('Best accuracy found: ', np.sqrt(np.abs(randomized_mse.best_score_)))
    return randomized_mse.best_params_
        
              
        
def DictionaryPlot(DictList, ChartName):
    """
    Creates bar plots using a dictionary input.

    Parameters
    ----------
    DictList : Dict.
        Dictionary containing numerical values as its values, to be plotted.
    ChartName : Str.
        Title of the chart.

    Returns
    -------
    None.

    """
    plt.figure(figsize = (20, 20))
    plt.bar(range(len(DictList)),DictList.values()) 
    plt.title(ChartName)
    plt.ylabel('Relevance')
    plt.xticks(ticks = range(len(DictList)), labels = list(DictList.keys()), rotation=90)
    plt.show()
  
def ConfusionMatrixPlot(ConfusionResults, ListFeatures, ListCoeffs):
    fig = sns.heatmap(ConfusionResults, annot =True, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Model: XGBoost \n Features: {}".format(dict(zip(ListFeatures,ListCoeffs))))
    plt.plot(fig)
    plt.show()
    
def ConfusionPairPlot(X_test, Y_test, model):
    """
    Creates a pair plot that colors the datapoints depending on if the points are True/False Positives/Negatives. This can be used to visualise the boundary cuts for the predictions.

    Parameters
    ----------
    X_test : Pandas DataFrame
        The dataframe sohuld contain the features required to perform predictions with the model.
    Y_test : One dimensional array
        Array containing the labels of the dataset.
    model : XGB model
        Trained classifier model

    Returns
    -------
    Pairplot.

    """
    try:
        y_pred = model.predict(X_test)
    except:
        print('Model does not work with a dataset, trying to use a DMatrix.')
        dtrain = xgb.DMatrix(X_test,Y_test)
        y_pred = model.predict(dtrain)
    Predictions = [round(value) for value in y_pred]
    X_dev = X_test.copy()
    for i in range(X_dev.shape[0]):
        if Predictions[i] == 0 and Y_test.iloc[i] == 0:
           Predictions[i] = 'True Negative'
        elif Predictions[i] == 0 and Y_test.iloc[i] == 1:
            Predictions[i] = 'False Negative'
        elif Predictions[i] == 1 and Y_test.iloc[i] == 0:
            Predictions[i] = 'False positive'
        elif Predictions[i] == 1 and Y_test.iloc[i] == 1:
            Predictions[i] = 'True positive'
    
    X_dev['Class'] = Predictions
    try:
        Feature_Plots_PCA.FeaturePlots(X_dev, 'Class')
    except: 
        Feature_Plots_PCA.FeaturePlots(X_dev.drop('PRI_jets',axis=1), 'Class')
    #Feature_Plots_PCA.PCAAnalysis(X_dev, 'Class')

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    wtrain *= sum_weight / sum(wtrain)
    wtest *= sum_weight / sum(wtest)
    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    return (dtrain, dtest, param)
    
def XGBoostersConfusionMatrix(DataSet, Y, paramList):
    # split data into train and test sets
    num_round = 200
    seed = 7
    test_size = 0.33
    try:
        DataSet.drop(['Label'],axis=1,inplace = True)
        DataSet.drop(['EventID'],axis=1,inplace = True)
        print('EventID and Label removed from DataSet.')
    except:
        print('DataSet already cleaned.')
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    dtrain = xgb.DMatrix(X_train.drop('Events_weight',axis =1), label=y_train, weight=X_train.Events_weight)
    dtest = xgb.DMatrix(X_test.drop('Events_weight',axis =1), label=y_test, weight=X_test.Events_weight )
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    sum_wpos = sum( X_train.Events_weight.iloc[i] for i in range(len(y_train)) if y_train.iloc[i] == 1.0  )
    sum_wneg = sum( X_train.Events_weight.iloc[i] for i in range(len(y_train)) if y_train.iloc[i] == 0.0  )
    #sum_wpos = len(y_train[y_train == 1])
    #sum_wneg = len(y_train[y_train == 0])
    print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))
  
    paramList['eval_metric'] = ['auc','ams@0']
    paramList['tree_method'] ='gpu_hist'
    paramList['ojective'] = 'binary:logistic'
    watchlist = [(dtest,'eval'), (dtrain,'train')]
    print ('loading data end, start to boost trees')
    Scalinglist = [0.001]
    for i in Scalinglist:
        paramList['scale_pos_weight'] = sum_wneg/sum_wpos * i
        Model = xgb.train(paramList, dtrain = dtrain,num_boost_round=num_round,evals = watchlist, early_stopping_rounds= 50, verbose_eval= False)
        print(Model)
        Matrix = xgb.DMatrix(DataSet.drop('Events_weight',axis=1),label=Y,weight=DataSet.Events_weight)
        y_pred = Model.predict(Matrix)
        predictions = y_pred > 0.5
        accuracy = accuracy_score(Y, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        GXBoost_confusion = confusion_matrix(Y,predictions,normalize=None)
        print('{} events misclassified as true with an ams score of {}, with weightings shifted by {}'.format(GXBoost_confusion[0,1], Model.best_score,i))
        
    sns.heatmap(GXBoost_confusion,annot=True)
    plt.xlabel('Predicted value')
    plt.ylabel('True value')
    plt.title("XGBoost")
    plt.show()
    fig, ax = plt.subplots(figsize=(300, 300))
    plot_tree(Model, num_trees=0, ax=ax, rankdir = 'LR')
    plt.show()
    for item in ['weight', 'gain', 'cover']:
        xgb.plot_importance(Model, importance_type = item, title = 'Feature importance: {}'.format(item))
        plt.show()
    
    ConfusionPairPlot(DataSet.drop('Events_weight',axis=1), Y, Model)
    #Model.get_booster().dump_model('XGBoost_model.txt', with_stats = True)
    Model.save_model('XGBoostModelFile')
    #GXBoost_coeff = dict(zip(DataSet.columns,
    #                     np.round(np.concatenate((Model.feature_importances_), axis=None), 3)))
    #print('GXBoost coefficients:{}'.format(GXBoost_coeff))
    #plt.figure(figsize = (20, 20))
    #plt.bar(range(len(GXBoost_coeff)),GXBoost_coeff.values()) 
    #plt.title("GXBoost")
    #plt.ylabel('Relevance')
    #plt.xticks(ticks = range(len(GXBoost_coeff)), labels = list(GXBoost_coeff.keys()), rotation=90)
    #plt.show()
    return Model


def XGBoostersFeatureComparison(DataSet, Y):
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    results = pd.DataFrame(columns=['num_features','features','Accuracy', 'ConfusionMatrix'])
    print("Training models")
    paramList = {'max_depth' : 6,
                     'nthread' : -1,
                     'tree_method' : 'gpu_hist',
                     'ojective' : 'binary:logistic',
                     'base_score' : 0.2,
                     'alpha' : 1 }
    model = XGBClassifier(**paramList)
    if len(np.unique(Y)) == 2:
      for k in range(1, X_train.shape[1] + 1):
        for subset in tqdm(itertools.combinations(range(X_train.shape[1]), k), leave = None):
            subset = list(subset)
            model.fit(X = X_train[X_train.columns[subset]], y = y_train, eval_metric = "ams@0" ,eval_set = [(X_test[X_test.columns[subset]], y_test)], verbose = False, early_stopping_rounds = 10)
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
                model.fit(X = X_train, y = y_train, eval_metric = "ams@0" ,eval_set = [(X_test, y_test)], verbose = True, early_stopping_rounds = 10)   
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
            if all(results['coeffs'][x]) != 0:
                ConfusionMatrixPlot(results['ConfusionMatrix'][x], results['features'][x], results['coeffs'][x])
                
            #DictionaryPlot(dict(zip(results['features'][x].tolist(),results['coeffs'][x].tolist())), 'Logistic Linear Regression with accuracy {}'.format(results['Accuracy'][x]))
            #print('Features for top result :{}'.format(dict(zip(results['features'][x].tolist(),results['coeffs'][x].tolist()))))
            for i in range(len(results['features'][x])):
                if results['coeffs'][x][i] != 0:
                   Features[results['features'][x][i]] = Features[results['features'][x][i]] + 1
                
    Features = {k: v for k, v in sorted(Features.items(), key = lambda item: item[1], reverse = True)}  
    print(Features)  
    DictionaryPlot(Features, "Frequency of features in best models for XGBoost")
    return results 
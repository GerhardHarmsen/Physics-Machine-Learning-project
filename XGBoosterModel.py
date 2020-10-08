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
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import Feature_Plots_PCA


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

class TreeModel():
    def __init__(self,DataSet,paramList = None):
        try:
            self.df = DataSet.drop(['EventID','Label'],axis=1)
            self.Y = DataSet.Label
            self.labels = self.df.drop('Events_weight',axis=1).columns
        except:
            try:
                self.df = DataSet.drop(['Label'],axis=1)
                self.Y = DataSet.Label
                self.labels = self.df.drop('Events_weight',axis=1).columns  
            except:
                print('DataSet not of the correct form. Ensure that the particle'
                  / 'dataset has been produced by the ConvertLHEToTxt module')
        
        if paramList == None:
            self.HyperParameters = {}
        else:
            self.HyperParameters = paramList
            
        self.SubSampleData()
            
    def SubSampleData(self):
        test_size = 0.3
        seed = 0
        X_train, X_test, y_train, y_test = train_test_split(self.df, self.Y,
                                                            test_size=test_size, 
                                                            random_state=seed)
        #X = pd.concat([X_train,y_train],axis=1)
        #BackGround = X[X.Label == 0]
        #Signal = X[X.Label == 1]
        #UpSampledData = resample( Signal,
        #                          replace = True,
        #                          n_samples =len(BackGround),
        #                          random_state = seed)
        #
        #self.TrainingData = pd.concat([BackGround,UpSampledData])
        #self.TestingData = pd.concat([X_test,y_test],axis = 1)
        self.TrainingData = pd.concat([X_train,y_train],axis =1)
        self.TestingData = pd.concat([X_test,y_test],axis=1)

    def HyperParameterTuning(self, NoofTests = 200):
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
        param_grid =  { 'learning_rate' : [0.01, 0.1, 0.5, 0.9],
                       'n_estimators' : [10, 50, 100, 150, 200],
                       'subsample': [0.3, 0.5, 0.9],
                       'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'base_score' : [0.1, 0.5, 0.9],
                       'reg_alpha' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                       'min_split_loss' : [0, 0.5, 0.8, 1, 2],
                       'reg_gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ],
                       'min_child_weight' : range(5,8)}
    
        model = XGBClassifier(objective = "binary:logistic")
        randomized_mse = RandomizedSearchCV(estimator = model, 
                                            param_distributions=param_grid,
                                            n_iter = NoofTests, scoring='roc_auc', 
                                            n_jobs =-1, cv=4, verbose = 1)
        randomized_mse.fit(self.TrainingData.drop(['Events_weight','Label'],axis=1), self.TrainingData.Label)
        print('Best parameters found: ', randomized_mse.best_params_)
        print('Best accuracy found: ', np.sqrt(np.abs(randomized_mse.best_score_)))
        self.HyperParameterResults = randomized_mse
        self.HyperParameters = randomized_mse.best_params_
        return randomized_mse.best_params_
        
              
        

    
    def ConfusionPairPlot(self,X_test,Y_test):
        """
        Creates a pair plot that colors the datapoints depending on if the points are True/False Positives/Negatives. This can be used to visualise the boundary cuts for the predictions.
        
        Parameters
        ----------
        X_test : Pandas DataFrame
            The dataframe sohuld contain the features required to perform predictions with the model.
        Y_test : One dimensional array
            Array containing the labels of the dataset.
        
        Returns
        -------
        Pairplot.
        
        """
        try:
            y_pred = self.Model.predict(X_test)
        except:
            print('Model does not work with a dataset, trying to use a DMatrix.')
            dtrain = xgb.DMatrix(X_test,Y_test)
            y_pred = self.Model.predict(dtrain)
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
    
    def TreeDiagram(self):
        fig, ax = plt.subplots(figsize=(300, 300))
        plot_tree(self.Model, num_trees=0, ax=ax, rankdir = 'LR')
        plt.show()
        for item in ['weight', 'gain', 'cover']:
            xgb.plot_importance(self.Model,
                                importance_type = item, 
                                title = 'Feature importance: {}'.format(item))
            plt.show()
    
    def XGBoostTrain(self):
        # split data into train and test sets
        num_round = 200
        dtrain = xgb.DMatrix(self.TrainingData.drop(['Events_weight','Label'],axis =1),
                             label=self.TrainingData.Label,
                             weight=self.TrainingData.Events_weight)
        dtest = xgb.DMatrix(self.TestingData.drop(['Events_weight','Label'],axis =1),
                            label=self.TestingData.Label,
                            weight=self.TestingData.Events_weight )
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        sum_wpos = sum( self.TrainingData.Events_weight.iloc[i] for i in range(len(self.TrainingData)) if self.TrainingData.Label.iloc[i] == 1.0  )
        sum_wneg = sum( self.TrainingData.Events_weight.iloc[i] for i in range(len(self.TrainingData)) if self.TrainingData.Label.iloc[i] == 0.0  )
        print ('Weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))
        paramList = self.HyperParameters
        paramList['eval_metric'] = ['auc','ams@0']
        paramList['tree_method'] ='gpu_hist'
        paramList['ojective'] = 'binary:logistic'
        watchlist = [(dtest,'eval'), (dtrain,'train')]
        print ('loading data end, start to boost trees')
        AdjustWeights = [0,0.001,0.01,0.1,1]
        AMSResults = []
        for i in AdjustWeights:
            paramList['scale_pos_weight'] = sum_wneg/sum_wpos * i
            self.Model = xgb.train(paramList, dtrain = dtrain,num_boost_round=num_round,evals = watchlist, early_stopping_rounds= 50, verbose_eval= False)
            AMSResults.append(self.Model.best_score)
        paramList['scale_pos_weight'] = sum_wneg/sum_wpos * AdjustWeights[AMSResults.index(max(AMSResults))]
        self.Model = xgb.train(paramList, dtrain = dtrain,num_boost_round=num_round,evals = watchlist, early_stopping_rounds= 50, verbose_eval= False)
        self.ModelPredictions(self.TestingData.drop(['Label'],axis=1),self.TestingData.Label)
        self.Model.save_model('XGBoostModelFile')
        self.TreeDiagram()
        self.ConfusionPairPlot(self.TestingData.drop(['Events_weight','Label'],axis=1), self.TestingData.Label)
        return self.Model
    
    def ModelPredictions(self,X_test,y_test):
        Matrix = xgb.DMatrix(X_test.drop('Events_weight',axis=1),label=y_test,weight=X_test.Events_weight)
        y_pred = self.Model.predict(Matrix)
        predictions = y_pred > 0.5
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        GXBoost_confusion = confusion_matrix(y_test,predictions,normalize=None)
        print('{} events misclassified as true with an ams score of {}'.format(GXBoost_confusion[0,1], self.Model.best_score))
        sns.heatmap(GXBoost_confusion,annot=True)
        plt.xlabel('Predicted value')
        plt.ylabel('True value')
        plt.title("XGBoost")
        plt.show()
        
    def AMSScore(self,DataSet,Y):
        dtrain = xgb.DMatrix(DataSet.drop('Events_weight',axis=1),Y)
        y_pred = self.Model.predict(dtrain)
        Predictions = [round(value) for value in y_pred]
        s = []
        b = []
        for i in range(y_pred.shape[0]):
            if Predictions[i] == 1 and Y.iloc[i] == 1:
               s.append(DataSet.Events_weight.iloc[i])
            elif Predictions[i] == 1 and Y.iloc[i] == 0:
               b.append(DataSet.Events_weight.iloc[i])
    
        print('Number of correctly classified signal events {}. Number of misclassified background events {}'.format(len(s),len(b)))
        s = sum(s)
        b = sum(b)
        print('The AMS score is {}'.format(np.sqrt(2*(s + b)*np.log(1 + s/b)-s)))
        
        
        

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
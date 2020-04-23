# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:43:45 2020

@author: gerha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import click
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LarsCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

import warnings
warnings.filterwarnings("ignore")

def DictionaryPlot(DictList, ChartName):
  plt.figure(figsize = (20, 20))
  plt.bar(range(len(DictList)),DictList.values()) 
  plt.title(ChartName)
  plt.ylabel('Relevance')
  plt.xticks(ticks = range(len(DictList)), labels = list(DictList.keys()), rotation=90)
  plt.show()
  
def ResultsLinearRegression(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    linreg_model = LinearRegression(normalize=True).fit(X_train, y_train)
    linreg_prediction = linreg_model.predict(X_test)
    linreg_mae = np.mean(np.abs(y_test - linreg_prediction))
    linreg_coefs = dict(
        zip(['Intercept'] + DataSet.columns.tolist(),
            np.round(np.concatenate((linreg_model.intercept_, linreg_model.coef_),
                                    axis= None), 3))
        )
    print('Linear Regression MAE: {}'.format(np.round(linreg_mae, 3)))
    print('Linear Regression coefficients:{}'.format(linreg_coefs))
    del linreg_coefs['Intercept']
    DictionaryPlot(linreg_coefs, 'Linear Regression')

#Subset regression analysis
def ResultsSubSetReg(DataSet, Y):
  X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
  if X_train.shape[1] > 16:
      if click.confirm('This will consider {} combinations of the orginal features. Are you sure you want to continue?'.format(2**X_train.shape[1] -2), default = False):
            print('Running Subset regression analysis')
            print('')
      else :
            return
          
  results = pd.DataFrame(columns=['num_features','features','MAE'])
  for k in range(1, X_train.shape[1] + 1):
      for subset in tqdm(itertools.combinations(range(X_train.shape[1]), k), leave = True):
          subset = list(subset)
          linreg_model = LinearRegression(normalize = True).fit(X_train[X_train.columns[subset]], y_train)
          linreg_prediction = linreg_model.predict(X_test[X_test.columns[subset]])
          linreg_mae = np.mean(np.abs(y_test - linreg_prediction))
          results = results.append(pd.DataFrame([{'num_features' : k,
                                                  'features' : subset,
                                                  'MAE' : linreg_mae}]))
  results = results.sort_values('MAE').reset_index()
  print(results.head())
  best_subset_model = LinearRegression(normalize=True).fit(X_train[:, results['features'][0]], y_train)
  best_subset_coefs = dict(
    zip(['Intercept'] + DataSet.columns.tolist(), 
        np.round(np.concatenate((best_subset_model.intercept_, best_subset_model.coef_), axis=None), 3))
  )
  print('Best Subset Regression MAE: {}'.format(np.round(results['MAE'][0], 3)))
  print('Best Subset Regression coefficients:{}'.format(best_subset_coefs))
  del best_subset_coefs['Intercept']
  DictionaryPlot(best_subset_coefs, 'Best Subset Regression')
#Ridge Regression

def ResultsRidgeRegression(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    ridge_cv = RidgeCV(normalize=True, alphas=np.logspace(-10, 1, 400))
    ridge_model = ridge_cv.fit(X_train, y_train)
    ridge_prediction = ridge_model.predict(X_test)
    ridge_mae = np.mean(np.abs(y_test - ridge_prediction))
    ridge_coefs = dict(zip(['Intercept'] + DataSet.columns.tolist(), 
        np.round(np.concatenate((ridge_model.intercept_, ridge_model.coef_), 
                                axis=None), 3))
                       )
    print('Ridge Regression MAE: {}'.format(np.round(ridge_mae, 3)))
    print('Ridge Regression coefficients:{}'.format(ridge_coefs))
    del ridge_coefs['Intercept']
    DictionaryPlot(ridge_coefs, 'Ridge Regression')
#LASSO

def ResultsLASSO(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    lasso_cv = LassoCV(normalize=True, alphas=np.logspace(-10, 1, 400))
    lasso_model = lasso_cv.fit(X_train, y_train)
    lasso_prediction = lasso_model.predict(X_test)
    lasso_mae = np.mean(np.abs(y_test - lasso_prediction))
    lasso_coefs = dict(zip(['Intercept'] + DataSet.columns.tolist(), 
        np.round(np.concatenate((lasso_model.intercept_, lasso_model.coef_), axis=None), 3))
                       )
    print('LASSO MAE: {}'.format(np.round(lasso_mae, 3)))
    print('LASSO coefficients:{}'.format(lasso_coefs))
    del lasso_coefs['Intercept']
    DictionaryPlot(lasso_coefs, 'LASSO')
#Elastic Net

def ResultsElasticNet(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    elastic_net_cv = ElasticNetCV(normalize=True, alphas=np.logspace(-10, 1, 400), 
                              l1_ratio=np.linspace(0, 1, 100))
    elastic_net_model = elastic_net_cv.fit(X_train, y_train)
    elastic_net_prediction = elastic_net_model.predict(X_test)
    elastic_net_mae = np.mean(np.abs(y_test - elastic_net_prediction))
    elastic_net_coefs = dict(zip(['Intercept'] + DataSet.columns.tolist(), 
        np.round(np.concatenate((elastic_net_model.intercept_, 
                                 elastic_net_model.coef_), axis=None), 3))
                             )
    print('Elastic Net MAE: {}'.format(np.round(elastic_net_mae, 3)))
    print('Elastic Net coefficients:{}'.format(elastic_net_coefs))
    del elastic_net_coefs['Intercept']
    DictionaryPlot(elastic_net_coefs, 'Elastic Net')

#LARS
def ResultsLARS(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    LAR_cv = LarsCV(normalize=True)
    LAR_model = LAR_cv.fit(X_train, y_train)
    LAR_prediction = LAR_model.predict(X_test)
    LAR_mae = np.mean(np.abs(y_test - LAR_prediction))
    LAR_coefs = dict(zip(['Intercept'] + DataSet.columns.tolist(), 
        np.round(np.concatenate((LAR_model.intercept_, LAR_model.coef_), axis=None), 3))
                     )
    print('Least Angle Regression MAE: {}'.format(np.round(LAR_mae, 3)))
    print('Least Angle Regression coefficients:{}'.format(LAR_coefs))
    del LAR_coefs['Intercept']
    DictionaryPlot(LAR_coefs, 'Least Angle Regression')

#Principal Component Regression 
def ResultsPCA(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    regression_model = LinearRegression(normalize=True)
    pca_model = PCA()
    pipe = Pipeline(steps=[('pca', pca_model), ('least_squares', regression_model)])
    param_grid = {'pca__n_components': range(1, 9)}
    search = GridSearchCV(pipe, param_grid)
    pcareg_model = search.fit(X_train, y_train)
    pcareg_prediction = pcareg_model.predict(X_test)
    pcareg_mae = np.mean(np.abs(y_test - pcareg_prediction))
    n_comp = list(pcareg_model.best_params_.values())[0]
    pcareg_coefs = dict(zip(['Intercept'] + ['PCA_comp_' + str(x) for x in range(1, n_comp + 1)], 
       np.round(np.concatenate((pcareg_model.best_estimator_.steps[1][1].intercept_, 
                                pcareg_model.best_estimator_.steps[1][1].coef_), axis=None), 3))
                        )
    print('Principal Components Regression MAE: {}'.format(np.round(pcareg_mae, 3)))
    print('Principal Components Regression coefficients:{}'.format(pcareg_coefs))
    del pcareg_coefs['Intercept']
    DictionaryPlot(pcareg_coefs, 'Principal Components Regression')

#Partial Least Sqaures Regression 
def ResultsPartialLeastSquares(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    pls_model_setup = PLSRegression(scale=True)
    param_grid = {'n_components': range(1, 9)}
    search = GridSearchCV(pls_model_setup, param_grid)
    pls_model = search.fit(X_train, y_train)
    pls_prediction = pls_model.predict(X_test)
    pls_mae = np.mean(np.abs(np.array(y_test) - pls_prediction))
    pls_coefs = dict(zip(DataSet.columns.tolist(), 
      np.round(np.concatenate((pls_model.best_estimator_.coef_), axis=None), 3))
                     )
    print('Partial Least Squares Regression MAE: {}'.format(np.round(pls_mae, 3)))
    print('Partial Least Squares Regression coefficients:{}'.format(pls_coefs))
    DictionaryPlot(pls_coefs, 'Partial Least Squares Regression')

def ResultsLogisticRegression(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    logreg_model = LogisticRegression(fit_intercept = False)
    logreg_model.fit(X_train, y_train)
    logreg_prediction = logreg_model.predict(X_test)
    logreg_confusion = plot_confusion_matrix(logreg_model, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize = 'true')
    logreg_coeff = dict(zip(DataSet.columns.tolist(),
                        np.round(np.concatenate((logreg_model.coef_), axis=None), 3)))
    print('Logistic Linear Regression coefficients:{}'.format(logreg_coeff))
    DictionaryPlot(logreg_coeff, 'Logistic Linear Regression')

def ResultsRFE(DataSet, Y):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, train_size = 0.75)
    logreg_model = LogisticRegression(fit_intercept = False)
    logreg_model.fit(X_train, y_train)
    RFE_model = RFE(logreg_model, n_features_to_select = 1)
    RFE_model.fit(X_train, y_train)
    RFE_Rank = RFE_model.ranking_
    featureRanking = dict(zip(DataSet.columns.tolist(),RFE_Rank))
    RFE_Prediction = RFE_model.predict(X_test)
    RFE_confusion = plot_confusion_matrix(RFE_model, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize = 'true')
    print("RFE Ranking of features:{}".format(featureRanking))
    DictionaryPlot(featureRanking, 'RFE model')
    
def RunAllLinearModels(DataSet, Y):  #Y is the feature that we are trying to predict using the other features
    ResultsLinearRegression(DataSet, Y)
    ResultsSubSetReg(DataSet, Y)
    ResultsRidgeRegression(DataSet, Y)
    ResultsLASSO(DataSet, Y)
    ResultsElasticNet(DataSet, Y)
    ResultsLARS(DataSet, Y)
    ResultsPCA(DataSet, Y)
    ResultsPartialLeastSquares(DataSet, Y)
    
def RunShrinkageForwJetDataBase():
    DataSet = pd.read_csv(r"C:\Users\gerha\Google Drive\Research\Post Doc\Physics\ML TTBar\wJetPsuedoRapidityDataSet.csv")
    DataSet = DataSet.replace(to_replace = [np.inf, -np.inf], value = np.nan)
    DataSet = DataSet.dropna()
    DataSet = DataSet[DataSet.IST == 1]
    print(DataSet)
    DataSet = DataSet.drop(columns = ["EventID", "IST", "MOTH1", "MOTH2", "VTIM", "SPIN", "ICOL1", "ICOL2"], axis = 1)
    print(DataSet.head())
    DataSet = DataSet.drop(columns = "P3", axis = 1)
    print(DataSet.head())
    Y = DataSet.PGD
    Y = Y.replace(to_replace = [-22, -21, -15, -13, -12, -11, -5, 5, 11, 12, 13, 15, 21, 22], value = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    DataSet = DataSet.drop(columns = "PGD", axis = 1)
    print(DataSet.head())
    print(Y.head())
    ResultsLogisticRegression(DataSet, Y)
    ResultsRFE(DataSet, Y)
    
def RunShrinkageForwTauDataBase():
    DataSet = pd.read_csv(r"C:\Users\gerha\Google Drive\Research\Post Doc\Physics\Tau DataSet\PsuedoRapidityDataSet.csv")
    DataSet = DataSet.replace(to_replace = [np.inf, -np.inf], value = np.nan)
    DataSet = DataSet.dropna()
    DataSet = DataSet[DataSet.IST == 1]
    print(DataSet)
    DataSet = DataSet.drop(columns = ["EventID", "IST", "MOTH1", "MOTH2", "VTIM", "SPIN", "ICOL1", "ICOL2"], axis = 1)
    print(DataSet.head())
    DataSet = DataSet.drop(columns = "P3", axis = 1)
    print(DataSet.head())
    Y = DataSet.PGD
    Y = Y.replace(to_replace = [-23, -15, -13, -11, 11, 13, 15, 23], value = [0, 1, 0, 0, 0, 0, 1, 0])
    DataSet = DataSet.drop(columns = "PGD", axis = 1)
    print(DataSet.head())
    print(Y.head())
    ResultsLogisticRegression(DataSet, Y)
    ResultsRFE(DataSet, Y)
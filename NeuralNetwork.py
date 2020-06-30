# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:01:04 2020

@author: gerha
"""


# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import talos as ta

from ann_visualizer.visualize import ann_viz;
# load the dataset

params = {'lr': 6,
     'first_neuron': 8,
     'hidden_layers': 1,
     'epochs': 5,
     'weight_regulizer': None,
     'emb_output_dims': None,
     'shape':'brick',
     'optimizer': 'Adam',
     'losses': 'binary_crossentropy',
     'activation': 'relu',
     'last_activation': 'sigmoid'}
    
# split into input (X) and output (y) variables
def NeuralNetwork(x_train, y_train, x_val, y_val, params):
    # define the keras model
    # first hidden layer had 12 nodes, second hiden layer had 8 nodes
    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))
   
    #model.add(Dropout(params['dropout']))
    #Add hidden layers
    HiddenNeurons = [params['first_neuron'] // 2]
    for items in range(params['hidden_layers']):
        HiddenNeurons.append(HiddenNeurons[-1] // 2)
      
    for i in range(params['hidden_layers']):
        model.add(Dense(HiddenNeurons[i], activation = params['activation'])) 
    
    
    model.add(Dense(1, 
                    activation=params['last_activation'],
                    kernel_initializer='normal'))
    
    # compile the keras model
    model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'],
                  metrics=['acc'])
      
      
    # fit the keras model on the dataset
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        epochs=params['epochs'],
                        verbose=0)
    # evaluate the keras model
    print(model.summary())
    return history, model

def NeuralNetScan(DataSet, Y):
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    p = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[0, 1, 2],
     'epochs':  [5],
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick'],
     'optimizer': ['Adam'],
     'losses': ['binary_crossentropy'],
     'activation': ['relu', 'tanh'],
     'last_activation': ['sigmoid']}
    t = ta.Scan(x=X_train,
            y=y_train,
            x_val=X_test,
            y_val=y_test,
            model=NeuralNetwork,
            params=p,
            experiment_name = 'Particle_DataSet')
    BestParameters = t.data.sort_values(by = 'val_acc', ascending = False).iloc[0]
    return t, BestParameters


def TrainedNeuralNetwork(DataSet, Y):
    params = {'lr': 3.2,
     'first_neuron': 8,
     'hidden_layers': 2,
     'epochs': 5,
     'weight_regulizer': None,
     'emb_output_dims': None,
     'shape':'brick',
     'optimizer': 'Adam',
     'losses': 'binary_crossentropy',
     'activation': 'relu',
     'last_activation': 'sigmoid'}
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    History, Model =  NeuralNetwork(X_train, y_train, X_test, y_test, params)   
    
    #ann_viz(Model, title="My first neural network", view = True)
    
    _, accuracy = Model.evaluate(X_test, y_test)
    Y_pred = Model.predict(X_test)
    y_pred = Y_pred>0.5 
    confuse = confusion_matrix(y_test,y_pred, normalize = 'true')
    fig = plt.figure()
    sns.heatmap(confuse, annot = True)
    plt.show()
    #Plot reponse curves for the neural netwok
    X = DataSet
    for column in DataSet.columns:
        X_design = X.copy()
        X_design = pd.DataFrame(X_design.median()).transpose()
        min_val = min(X.loc[:, column])
        max_val = max(X.loc[:, column])
        seq = np.linspace(start=min_val, stop=max_val, num=50)
        to_predict = []
        for results in seq:
            X_design.loc[0,column] = results
            to_predict.append(X_design.copy())
        
        to_predict = pd.concat(to_predict)
        to_predict = sc.transform(to_predict)
        predictions = Model.predict(to_predict)
        #predictions = predictions > 0.5
        
        plt.plot(seq,predictions)
        plt.xlabel(column)
        plt.ylabel("Class")
        plt.title("Response vs {}".format(column))
        plt.show()
    #TestingTheFeaturesofModel(model, DataSet1, Y)
    print('Accuracy: %.2f' % (accuracy*100))


def TestingTheFeaturesofModel(model, DataSet, Y):
    sc = StandardScaler()
    DataSet = pd.DataFrame(sc.fit_transform(DataSet), columns = DataSet.columns)
    Y_pred = model.predict(DataSet)
    y_pred = Y_pred>0.5 
    confuse = confusion_matrix(Y,y_pred, normalize = 'true')
    fig = plt.figure()
    plt.title('Unaltered features confusion plot')
    sns.heatmap(confuse, annot = True)
    plt.show()
    for Feature in DataSet.columns:
        Temp = DataSet
        Temp[Feature] =Temp[Feature] +  Temp[Feature].describe()[2]
        Y_pred = model.predict(Temp)
        y_pred = Y_pred>0.5 
        confuse = confusion_matrix(Y,y_pred, normalize = 'true')
        fig = plt.figure()
        plt.title('{} increased by 1 STD deviation.'.format(Feature))
        sns.heatmap(confuse, annot = True)
        plt.show()
        Temp[Feature] =Temp[Feature] -  Temp[Feature].describe()[2]
        Y_pred = model.predict(Temp)
        y_pred = Y_pred>0.5 
        confuse = confusion_matrix(Y,y_pred, normalize = 'true')
        fig = plt.figure()
        plt.title('{} decreased by 1 STD deviation.'.format(Feature))
        sns.heatmap(confuse, annot = True)
        plt.show()
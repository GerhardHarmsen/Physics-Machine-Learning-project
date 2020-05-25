# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:01:04 2020

@author: gerha
"""


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# load the dataset

# split into input (X) and output (y) variables
def NeuralNetwork(DataSet, Y):
    DataSet1 = DataSet
    sc = StandardScaler()
    DataSet = sc.fit_transform(DataSet)
    seed = 7
    test_size = 0.33
    Epochs = 10
    checkpoint_path = r"C:\Users\gerha\Google Drive\Research\Post Doc\Python Codes\NeuralNetSaves\cp.ckpt"
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=5)
    History = model.fit(X_train, y_train, epochs=Epochs, batch_size=10, callbacks=[cp_callback])
    # evaluate the keras model
    print(model.summary())
    fig1, fig2 = plt.subplots()
    fig1 = plt.plot(range(Epochs),History.history['loss'])
    fig2 = plt.plot(range(Epochs), History.history['accuracy'])
    plt.show()
    _, accuracy = model.evaluate(X_test, y_test)
    Y_pred = model.predict(X_test)
    y_pred = Y_pred>0.5 
    confuse = confusion_matrix(y_test,y_pred, normalize = 'true')
    fig = plt.figure()
    sns.heatmap(confuse, annot = True)
    plt.show()
    TestingTheFeaturesofModel(model, DataSet1, Y)
    print('Accuracy: %.2f' % (accuracy*100))

def TestingTheFeaturesofModel(model, DataSet, Y):
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
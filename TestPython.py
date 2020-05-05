# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:01:04 2020

@author: gerha
"""


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# load the dataset
DataSet = pd.read_csv(r"C:\Users\gerha\Google Drive\Research\Post Doc\Physics\ttBar\EventData.csv")
Y = DataSet.Label
DataSet = DataSet.drop(labels = 'Label', axis =1 )
DataSet = DataSet.drop(labels = 'EventID', axis =1)
# split into input (X) and output (y) variables
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=5, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
Y_pred = model.predict(X_test)
y_pred = Y_pred>0.5 


print('Accuracy: %.2f' % (accuracy*100))

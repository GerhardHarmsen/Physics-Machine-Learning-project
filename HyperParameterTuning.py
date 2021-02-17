# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:09:08 2021

@author: gerha
"""

from XGBoosterModel import TreeModel, DataCuts
from Feature_Plots_PCA import PCAPlotter
import Feature_Plots_PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import multiprocessing as mp


def HyperParameters(Smuon_Mass, Neutralino_Mass):
    HyperParameterResults = dict()    
    BackGroundData=pd.read_csv(r'/usr/src/app/Background_Events/EventData.csv')
    BackGroundData.drop('EventID',axis=1,inplace=True)
    
    SignalEvents = pd.read_csv('/usr/src/app/Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neatralino_{}/EventData.csv'.format(Smuon_Mass,Neutralino_Mass))
    SignalEvents.drop(['EventID'],axis=1,inplace=True)  
            
    DataSet = pd.concat([BackGroundData,SignalEvents])
    DataSet.sample(frac=1)
        
    DataSet = DataCuts(DataSet)
    
    XGBModel = TreeModel(DataSet,ApplyDataCut=False)
    
    
    #XGBModel.HyperParameterTuning(4)

    #return XGBModel.HyperParameters


    return {'Weights' : sum(DataSet.Events_weight[DataSet.Label == 1])}


def CodeToRun():
    NEUTRALINOMASS=[270, 220, 190, 140, 130, 140, 95, 80, 60, 60, 65, 55, 200, 190, 180, 195, 96, 195, 96]
    SMUONMASS=[360, 320, 290, 240, 240, 420, 500, 400, 510, 200, 210, 250, 450, 500, 400, 400, 400, 200, 200]
    
    AllDict = dict()
    
    for i in range(len(SMUONMASS)):
       AllDict['Smuon_Mass_{}_Neatralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = HyperParameters(SMUONMASS[i], NEUTRALINOMASS[i])
       
    print(AllDict)
    with open(r'/usr/src/app/DictionaryWeights.json', 'w') as json_file:
        json.dump(AllDict, json_file)
        
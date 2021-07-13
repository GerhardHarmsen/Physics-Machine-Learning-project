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
import os
import tqdm

def HyperParameters(Smuon_Mass, Neutralino_Mass,SignalEventCSV,BackgroundCSV,NoofTests = 400,Noof_jobs = -1):
    HyperParameterResults = dict()    
    BackGroundData=pd.read_csv(os.path.join(BackgroundCSV, 'EventData.csv'))
    BackGroundData.drop('EventID',axis=1,inplace=True)
    
    SignalEventPath = os.path.join(SignalEventCSV,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(Smuon_Mass,Neutralino_Mass))
    SignalEvents = pd.read_csv(os.path.join(SignalEventPath,'EventData.csv'))
    SignalEvents.drop(['EventID'],axis=1,inplace=True)  
            
    DataSet = pd.concat([BackGroundData,SignalEvents])
    DataSet.sample(frac=1)
        
    DataSet = DataCuts(DataSet)
    
    XGBModel = TreeModel(DataSet,ApplyDataCut=False)
    
    
    XGBModel.HyperParameterTuning(NoofTests,Noof_jobs,SignalEventPath)

    HyperParameterResults = XGBModel.HyperParameters
    
    with open(os.path.join(SignalEventPath,'HyperparameterDictionary.json'), 'w') as json_file:
        json.dump(HyperParameterResults, json_file)
        
    
    
    


def CombineJSON(SignalEventCSV,JSONSaveFolder):
    #### SMuon Neutralino mass cases
    NEUTRALINOMASS=[270, 220, 190, 140, 130, 140, 95, 80, 60, 60, 65, 55, 200, 190, 180, 195, 96, 195, 96, 175, 87, 125, 100, 70, 100, 68, 120, 150, 75, 300, 500, 440, 260]
    SMUONMASS=[360, 320, 290, 240, 240, 420, 500, 400, 510, 200, 210, 250, 450, 500, 400, 400, 400, 200, 200, 350, 350, 375, 260, 350, 300, 275, 475, 300, 450, 310, 510, 450, 275]
      
    AllDict = dict()
    
    for i in tqdm.tqdm(range(len(SMUONMASS))):
        SignalEventPath = os.path.join(SignalEventCSV,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
        try: 
            with open(os.path.join(SignalEventPath,'HyperparameterDictionary.json'), 'r') as json_file:
                AllDict['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = json.load(json_file)
        except:
            print('Case of ({},{}) not found'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
       
    with open(os.path.join(JSONSaveFolder,'HyperparameterDictionary.json'), 'w') as json_file:
        json.dump(AllDict, json_file)
        

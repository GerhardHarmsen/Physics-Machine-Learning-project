# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:58:43 2021

@author: gerha
"""

from XGBoosterModel import TreeModel, DataCuts
from Feature_Plots_PCA import PCAPlotter
import Feature_Plots_PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def Pipeline(DataSet, paramList = None):
    DataSet = DataCuts(DataSet)
    DataSet.drop(['EventID'],axis=1,inplace=True)
 
    PCAPlots = PCAPlotter(DataSet,'Label')
    PCAPlots.PCAAnalysis()
    
    if paramList == None:
        XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False) 
        XGBModel.HyperParameterTuning()
    else:
        XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False, paramList=paramList) 
        
    XGBModel.XGBoostTrain()
    XGBModel.SHAPValuePlots()
    
    PCAMag = {}
    for items in PCAPlots.FeaturePCAValues['Leptons 2 Jets 2']:
        PCAMag[items] = np.sqrt(sum(abs(PCAPlots.FeaturePCAValues['Leptons 2 Jets 2'][items])))
    PCAMag.pop('PRI_nleps')
    PCAMag.pop('PRI_jets')
    PCAMag = dict(sorted(PCAMag.items(), key=lambda item: item[1]))

    DropColumns = list(PCAMag.keys())[:8]
    print(DropColumns) 

    DataSet.drop(DropColumns,axis=1,inplace=True)
    #DataSet.drop('DER_PT_subleading_lepton_ratio_PT_leading_jet',axis=1,inplace = True)
    
    PCAPlots = PCAPlotter(DataSet,'Label')
    PCAPlots.PCAAnalysis()
    
    if paramList == None:
        XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False) 
        XGBModel.HyperParameterTuning()
    else:
        XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False, paramList=paramList) 
        
    XGBModel.XGBoostTrain()
    XGBModel.SHAPValuePlots()
    

if "__main__":
    Smuon_200_Neutranlino_96 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_96\EventData.csv'
    Smuon_200_Neutralino_195 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_195\EventData.csv'
    Smuon_400_Neutralino_96 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv'
    Smuon_400_Neutralino_195 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_195\EventData.csv'
    DataSetList = [Smuon_200_Neutranlino_96, Smuon_200_Neutralino_195, Smuon_400_Neutralino_96, Smuon_400_Neutralino_195] 

    paramList = {'subsample': 1,
                 'reg_gamma': 0.4,
                 'reg_alpha': 0.1,
                 'n_estimators': 200,
                 'min_split_loss': 2,
                 'min_child_weight': 5,
                 'max_depth': 5,
                 'learning_rate': 0.1}
    
    Pipeline(pd.read_csv(Smuon_400_Neutralino_195), paramList = paramList)
    #for items in DataSetList:
    #    Pipeline(pd.read_csv(items), paramList = paramList)
    
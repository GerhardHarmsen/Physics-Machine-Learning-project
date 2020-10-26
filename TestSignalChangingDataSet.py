# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:24:22 2020

@author: gerha
"""
from XGBoosterModel import TreeModel, DataCuts
import Feature_Plots_PCA
import pandas as pd

def RunTests():
    paramList = {'subsample': 1, 'reg_gamma': 0.4, 'reg_alpha': 0.1, 'n_estimators': 200, 'min_split_loss': 0.8, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'base_score': 0.9}

    TrainDataSet = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    XGBModel = TreeModel(TrainDataSet,paramList)
    XGBModel.XGBoostTrain()
    TestList = {}
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_96\EventData.csv')
    Y = TestDataSet1.Label
    XGBModel.ModelPredictions(TestDataSet1)
    TestList['Smuon_200_Neutralino_96'] = XGBModel.AMSScore(TestDataSet1)
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_195\EventData.csv')
    Y = TestDataSet1.Label
    XGBModel.ModelPredictions(TestDataSet1)
    TestList['Smuon_200_Neutralino_195'] = XGBModel.AMSScore(TestDataSet1)
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    Y = TestDataSet1.Label
    XGBModel.ModelPredictions(TestDataSet1)
    TestList['Smuon_400_Neutralino_96'] = XGBModel.AMSScore(TestDataSet1)
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_195\EventData.csv')
    Y = TestDataSet1.Label
    XGBModel.ModelPredictions(TestDataSet1)
    TestList['Smuon_400_Neutralino_195'] = XGBModel.AMSScore(TestDataSet1)
    print(TestList)
    
def RunPCAAnalysis():
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    Feature_Plots_PCA.PCAAnalysis(TestDataSet1,'Label')
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_195\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    Feature_Plots_PCA.PCAAnalysis(TestDataSet1,'Label')
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    TestDataSet1 = DataCuts(TestDataSet1)
    Feature_Plots_PCA.PCAAnalysis(TestDataSet1,'Label')
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_195\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    Feature_Plots_PCA.PCAAnalysis(TestDataSet1,'Label')

if __name__ is '__main__':
    RunTests()
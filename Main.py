# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#User defined python files
import ConvertLHEtoTxt
import Feature_Plots_PCA
import GXBoosterConfusionMAtrix
import Shrinkage_methods
import NeuralNetwork
#Python Modules
import pandas as pd
import numpy as np
import click


paramList = {'max_depth' : 6,
                     'nthread' : -1,
                     'tree_method' : 'gpu_hist',
                     'ojective' : 'binary:logistic',
                     'base_score' : 0.5,
                     'reg_alpha' : 0.1 }

if __name__ == '__main__':
    File = open('Description.txt', 'r')
    Lines = File.readlines()
    for line in Lines:
        print(line)
        
    File.close()

def Convert():
    SelectedDirectory = ConvertLHEtoTxt.RunAllconversions()
    print('All conversion completed files "LHEEventData.csv", "PsuedoRapidityDataSet.csv", "EventData.csv" added to folder: {}'.format(SelectedDirectory))
    return SelectedDirectory


def Analyse(DataSet, Labels, BlockFeatures = []):
    while True:
        try:
            DataSet = DataSet.drop(labels = BlockFeatures, axis = 1)

        except:
            print("The function needs a list of features to block in the analysis. To allow all features input '[]', otherwise enter '[feature1, feature2]'")
            BlockFeatures = input("Please include a list of features to block from the analysis.")  
            
        else:
            paramList = GXBoosterConfusionMAtrix.HyperParameterTuning()
            GXBoosterConfusionMAtrix.XGBoostersConfusionMatrix(DataSet, Labels, paramList)
            Shrinkage_methods.ResultsLogisticRegression(DataSet, Labels)
            LogisticResults = Shrinkage_methods.ResultsRFE(DataSet, Labels)
            XGBoostResults = GXBoosterConfusionMAtrix.XGBoostersFeatureComparison(DataSet, Labels)
            TALOSScanResults, BestResults = NeuralNetwork.NeuralNetScan(DataSet, Labels)
            print(BestResults)
            return LogisticResults, XGBoostResults, TALOSScanResults
            break

def TestForNanInDataSet(DataSet):
    COUNT = 0
    for i in range(DataSet.shape[0]):
        if DataSet.iloc[i].isnull().values.any(): COUNT += 1
    
    print('The dataset has {} events with nan values.'.format(COUNT))

    if COUNT > 0:
        if click.confirm('Nan values will prevent the analysis from completeing. Do you wish to remove all events with nan values?', default = False):
            print("Removed {} events.".format(COUNT))
            DataSet = DataSet.dropna()
            return DataSet, True
        else:   
            return DataSet, False
      
    else:
      return DataSet, True

def DataCuts(DataSet):
    """Function for introducing momentum and pseudorapidity cuts to the data """
    ####Clean the jet signals. To remove any soft jets.###
    DataSet0 = DataSet[DataSet.PRI_jets == 0]
    DataSet1 = DataSet[(DataSet.PRI_jets == 1) & (DataSet.PRI_leading_jet_pt >= 25) & (abs(DataSet.PRI_leading_jet_eta) <= 2.5)]
    DataSet2 = DataSet[(DataSet.PRI_jets >= 2) & (DataSet.PRI_leading_jet_pt >= 25) & (abs(DataSet.PRI_leading_jet_eta) <= 2.5) & (DataSet.PRI_subleading_jet_pt >= 25) & (abs(DataSet.PRI_subleading_jet_eta) <= 2.5)]
    print('{} events removed from the dataset'.format(len(DataSet)-len(pd.concat([DataSet0,DataSet1,DataSet2]))))
    Dataset = pd.concat([DataSet0,DataSet1,DataSet2])
    ### Clean the leptonic signals to remove any soft leptons####
    DataSet0 = DataSet[DataSet.PRI_nleps == 0]
    DataSet1 = DataSet[(DataSet.PRI_nleps == 1) & (DataSet.PRI_lep_leading_pt >= 10) & (abs(DataSet.PRI_lep_leading_eta) <= 2.5)]
    DataSet2 = DataSet[(DataSet.PRI_nleps >= 2) & (DataSet.PRI_lep_leading_pt >= 10) & (abs(DataSet.PRI_lep_leading_eta) <= 2.5) & (DataSet.PRI_lep_subleading_pt >= 10) & (abs(DataSet.PRI_lep_subleading_eta) <= 2.5)]
    DataSet = pd.concat([DataSet0,DataSet1,DataSet2])
    print('{} events removed from the dataset'.format(len(DataSet)-len(pd.concat([DataSet0,DataSet1,DataSet2]))))
    return DataSet


def PreprocessData(DataSet, Label = 'Label'):
    try :
        DataSet = DataSet.drop(labels = 'EventID', axis = 1)
    except:
        print('EventIDs already removed')
    try :
        DataSet = DataSet.drop(labels = 'Events_weight', axis = 1)
    except:
        print('Event_weights already removed')
        
    DataSet = DataCuts(DataSet)
    Feature_Plots_PCA.FeaturePlots(DataSet, Label)
    try: 
        DataSet.drop(labels = ['PRI_leading_jet_pt', 'PRI_subleading_jet_pt', 'PRI_leading_jet_eta', 
       'PRI_subleading_jet_eta'], axis =1, inplace = True)
        print('Removing jet related features')
    except:
        print('Unable to remove all jet related features')
    DataSet, FeatureTest = TestForNanInDataSet(DataSet)
    if FeatureTest:
        print("Performing feature analysis.")
        Feature_Plots_PCA.FeaturePlots(DataSet, 'Label')
        Feature_Plots_PCA.PCAAnalysis(DataSet, 'Label')
    
def ConvertAndAnalyse():
    SelectedDirectory = Convert()
        
    print('Extracting data from file EventData.csv')
    DataSet = pd.read_csv(SelectedDirectory + '/EventData.csv')
    DataSet = DataSet.drop(labels = ['EventID', 'Events_weight'], axis = 1)
    DataSet, FeatureTest = TestForNanInDataSet(DataSet)
    if FeatureTest:
        print("Performing feature analysis.")
        Feature_Plots_PCA.FeaturePlots(DataSet, 'Label')
        Feature_Plots_PCA.PCAAnalysis(DataSet, 'Label')
    print('Performing shrinkage analysis.')
    Y = DataSet.Label
    DataSet = DataSet.drop(labels = 'Label', axis = 1)
    print('Example of the dataset')
    print(DataSet.head())
    if len(Y[Y == 1])/len(Y) < 0.3 or len(Y[Y == 1])/len(Y) > 0.7:
        if click.confirm('The dataset contains {}% signal data do you wish to continue?'.format((len(Y[Y == 1])/len(Y))*100)):
            LogisticResults, XGBoostResults = Analyse(DataSet, Y, [])
            return LogisticResults, XGBoostResults
        
        else:
            pass
    else:
        print('Running analysis with {}% of dataset as signal.'.format((len(Y[Y == 1])/len(Y))*100))
        LogisticResults, XGBoostResults = Analyse(DataSet, Y, [])
        return LogisticResults, XGBoostResults
        
        
    
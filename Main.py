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
import click


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


def Analyse(DataSet, Labels, BlockFeatures):
    while True:
        try:
            DataSet = DataSet.drop(labels = BlockFeatures, axis = 1)

        except:
            print("The function needs a list of features to block in the analysis. To allow all features input '[]', otherwise enter '[feature1, feature2]'")
            BlockFeatures = input("Please include a list of features to block from the analysis.")  
            
        else:
            paramList = {'max_depth' : 6,
                     'nthread' : -1,
                     'tree_method' : 'gpu_hist',
                     'ojective' : 'binary:logistic',
                     'base_score' : 0.5,
                     'alpha' : 0.1 }
            DataSet, FeatureTest = TestForNanInDataSet(DataSet)
            GXBoosterConfusionMAtrix.XGBoostersConfusionMatrix(DataSet, Labels, paramList)
            Shrinkage_methods.ResultsLogisticRegression(DataSet, Labels)
            LogisticResults = Shrinkage_methods.ResultsRFE(DataSet, Labels)
            XGBoostResults = GXBoosterConfusionMAtrix.XGBoostersFeatureComparison(DataSet, Labels)
            TALOSScanResults, BestResults = NeuralNetwork.NeuralNetScan(DataSet, Y)
            print(BestResults)
            return LogisticResults, XGBoostResults, TALOSScanResults
            break

def TestForNanInDataSet(DataSet):
    RemovalList = []
    for column in DataSet.columns:
      if (len(DataSet[column]) - DataSet[column].count()) > 0: 
         print("{} has {} nan values.".format(column, len(DataSet[column]) - DataSet[column].count()))
         RemovalList.append(column)
         
         if click.confirm('Nan values will prevent the analysis from completeing. Do you wish to remove the above listed columns?', default = False):
            print("Removing selected feature columns.")
            DataSet = DataSet.drop(labels = RemovalList, axis = 1)
            return DataSet, True
         else:   
            return DataSet, False
      
      else:
        return DataSet, True
        
def ConvertAndAnalyse():
    SelectedDirectory = Convert()
        
    print('Extracting data from file EventData.csv')
    DataSet = pd.read_csv(SelectedDirectory + '/EventData.csv')
    DataSet = DataSet.drop(labels = 'EventID', axis = 1)
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
        
        
    
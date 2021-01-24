# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:24:22 2020

@author: gerha
"""
from XGBoosterModel import TreeModel, DataCuts, RemoveFeaturesNotinPaper
from Feature_Plots_PCA import PCAPlotter
import Feature_Plots_PCA
import pandas as pd

def ExtractSignalMasses(S):
    """
    Parameters
    ----------
    S : String
        Path to the datafile so that the containing Smuon_XX_Neutralino_XX.

    Returns
    -------
    String containing the Smuon and Neutranlino mass.

    """
    Start = S.find('Smuon')
    End = S.find('EventData') - 1
    return S[Start:End]

def RemoveFeatures(DataSet):
    ### The PCA values for the following features was too small ##########
    RemoveColumns = [ 'PRI_subleading_jet_eta', 'PRI_lep_leading_phi', 
                     'PRI_lep_leading_eta', 'PRI_leading_jet_eta', 
                     'PRI_lep_subleading_eta', 'PRI_lep_subleading_phi', 
                     'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 
                     'DER_PT_subleading_lepton_ratio_PT_leading_jet', 
                     'DER_P_T_ratio_lep_pair']
    
    #### These features were highly correlated to other chosen features#####
    RemoveColumns = RemoveColumns + ['PRI_leading_jet_pt', 'PRI_subleading_jet_pt', 
                          'DER_PT_leading_lepton_ratio_PT_leading_jet',
                          'DER_PT_leading_lept_ratio_HT', 
                          'DER_ST_ratio_PT_Leading_jet', 'DER_PT_subleading_ratio_HT']
    return DataSet.drop(RemoveColumns,axis=1)

def RunTests():
    Smuon_200_Neutranlino_96 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_96\EventData.csv'
    Smuon_200_Neutralino_195 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_195\EventData.csv'
    Smuon_400_Neutralino_96 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv'
    Smuon_400_Neutralino_195 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_195\EventData.csv'
    DataSetList = [Smuon_200_Neutranlino_96, Smuon_200_Neutralino_195, Smuon_400_Neutralino_96, Smuon_400_Neutralino_195] 
    
    paramList ={'subsample': 1, 'reg_gamma': 0.4, 'reg_alpha': 0.1, 'n_estimators': 200, 'min_split_loss': 2, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'base_score': 0.9}
    Results = []
    TestList = {}
    for items in DataSetList:
        TrainDataSet = pd.read_csv(items) 
        TrainDataSet = RemoveFeatures(TrainDataSet)
        XGBModel = TreeModel(TrainDataSet,paramList,SubSampleDataSet=True)
        XGBModel.XGBoostTrain()
        Start = DataSetList[0].find('Smuon')
        End = DataSetList[0].find('EventData') - 1
        DataSetList[0][Start:End]
        print('Model trained on DataSet with {}'.format(ExtractSignalMasses(items)))
        for Tests in DataSetList:
            print('Model tested on the dataset with {}'.format(ExtractSignalMasses(Tests)))
            TestDataSet = pd.read_csv(Tests)
            TestDataSet.drop('EventID',axis=1,inplace=True)
            Y = TestDataSet.Label
            XGBModel.ModelPredictions(TestDataSet)
            TestList[ExtractSignalMasses(Tests)] = XGBModel.AMSScore(TestDataSet)
        print('The AMS score for the model trained with the {} dataset'.format(ExtractSignalMasses(items)))        
        Results.append(TestList)
        print(TestList)
        print(Results)
    
    print(Results)        
   
def RunPCAAnalysis():
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    PCAPlots = PCAPlotter(TestDataSet1,'Label')
    PCAPlots.PCAAnalysis()
    ############################################################################
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_195\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    PCAPlots = PCAPlotter(TestDataSet1,'Label')
    PCAPlots.PCAAnalysis()
    ############################################################################
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    PCAPlots = PCAPlotter(TestDataSet1,'Label')
    PCAPlots.PCAAnalysis()
    ############################################################################
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_195\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    PCAPlots = PCAPlotter(TestDataSet1,'Label')
    PCAPlots.PCAAnalysis()
    
def FeaturePlots():
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    Feature_Plots_PCA.FeaturePlots(TestDataSet1, 'Label')
    ############################################################################
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_200_Neutralino_195\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    Feature_Plots_PCA.FeaturePlots(TestDataSet1, 'Label')
    ############################################################################
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    Feature_Plots_PCA.FeaturePlots(TestDataSet1, 'Label')
    ############################################################################
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_195\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1 = RemoveFeatures(TestDataSet1)
    TestDataSet1.drop(['EventID','Events_weight'],axis=1,inplace=True)
    Feature_Plots_PCA.FeaturePlots(TestDataSet1, 'Label')

if __name__ == '__main__':
    RunPCAAnalysis()
    #RunTests()
    #FeaturePlots()
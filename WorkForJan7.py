# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:18:53 2020

@author: gerha
"""
from XGBoosterModel import TreeModel, DataCuts
from Feature_Plots_PCA import PCAPlotter
import Feature_Plots_PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def TestOneFeature():
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    Columns = TestDataSet1.columns
    Columns = Columns.drop(['PRI_nleps','PRI_jets','Events_weight', 'Label', 'DER_ST_ratio_HT'])
    TestDataSet = TestDataSet1.drop(Columns,axis=1)
    PCAPlots = PCAPlotter(TestDataSet,'Label')
    PCAPlots.PCAAnalysis()
    print(PCAPlots.FeaturePCAValues['Leptons 2 Jets 2']['DER_ST_ratio_HT'])
    ST_HT_ratio_Percentage = [PCAPlots.FeaturePCAValues['Leptons 2 Jets 2']['DER_ST_ratio_HT']]
    AddedColumns = ['DER_ST_ratio_HT']
    while len(Columns) > 0:
        AddedColumns.append(Columns[0])
        Columns = Columns.drop(Columns[0])
        TestDataSet = TestDataSet1.drop(Columns,axis=1)
        PCAPlots = PCAPlotter(TestDataSet,'Label')
        PCAPlots.PCAAnalysis()
        print(PCAPlots.FeaturePCAValues['Leptons 2 Jets 2']['DER_ST_ratio_HT'])
        ST_HT_ratio_Percentage.append(PCAPlots.FeaturePCAValues['Leptons 2 Jets 2']['DER_ST_ratio_HT'])
        
    ST_HT_ratio_Percentage

    ax2 = plt.gca()
            
    X = np.arange(len(ST_HT_ratio_Percentage))
    width = 0.25
    List = [ST_HT_ratio_Percentage[i][0] for i in range(len(ST_HT_ratio_Percentage))]
    ax2.bar(X - width/2,List, width, color = 'b',label='PCA1')
    List = [ST_HT_ratio_Percentage[i][1] for i in range(len(ST_HT_ratio_Percentage))]
    ax2.bar(X + width/2,List, width, color = 'r',label='PCA2')
           
    ax2.set_ylabel('Percentage of PCA score')
    ax2.set_title('Percentage that each feature makes up of the PCA value')
    ax2.set_xlabel('Feature added in iteration')
    ax2.set_xticks(X)
    ax2.set_xticklabels(AddedColumns, rotation = 'vertical')
    ax2.legend()

def TestColumns(Feature = 'All', ShowPCAPlots = True):
    """
    This function tests returns the percentage of the contribution the the selected features contribute to the PCA values. The features provided are the ones checked against all the other columns.

    Parameters
    ----------
    Columns : String or list, optional
        DESCRIPTION. The default is 'All' which will sequentially tests all the features in the database. You can pass a list of features that you want to test or a single feature.

    Returns
    -------
    None.

    """
   
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    
    if Feature == 'All':
        TestColumns = TestDataSet1.columns
    else: 
        if type(Feature) == str:
            TestColumns = [Feature]
        elif type(Feature) == list:  TestColumns = Feature
        else: print('Feature needs to be of type string or list.')  
    for Column in TestColumns:
        Columns = TestDataSet1.columns
        Columns = Columns.drop(['PRI_nleps','PRI_jets','Events_weight', 'Label'] + [Column])
        TestDataSet = TestDataSet1.drop(Columns,axis=1)
        PCAPlots = PCAPlotter(TestDataSet,'Label')
        PCAPlots.PCAAnalysis( ShowPlots = ShowPCAPlots)
        print(PCAPlots.FeaturePCAValues['Leptons 2 Jets 2'][Column])
        Column_Percentage = [PCAPlots.FeaturePCAValues['Leptons 2 Jets 2'][Column]]
        AddedColumns = [Column]
        while len(Columns) > 0:
            AddedColumns.append(Columns[0])
            Columns = Columns.drop(Columns[0])
            TestDataSet = TestDataSet1.drop(Columns,axis=1)
            PCAPlots = PCAPlotter(TestDataSet,'Label')
            PCAPlots.PCAAnalysis(ShowPlots = ShowPCAPlots)
            print(PCAPlots.FeaturePCAPercentage['Leptons 2 Jets 2'][Column])
            Column_Percentage.append(PCAPlots.FeaturePCAPercentage['Leptons 2 Jets 2'][Column])
            
        Column_Percentage

        
                
        X = np.arange(len(Column_Percentage))
        Barplot = plt.figure()
        ax = Barplot.add_axes([0,0,1,1])
        width = 0.25
        List = [Column_Percentage[i][0] for i in range(len(Column_Percentage))]
        ax.bar(X - width/2,List, width, color = 'b',label='PCA1')
        List = [Column_Percentage[i][1] for i in range(len(Column_Percentage))]
        ax.bar(X + width/2,List, width, color = 'r',label='PCA2')
               
        ax.set_ylabel('Percentage of PCA score')
        ax.set_title('Percentage that each feature makes up of the PCA value starting with the {} feature'.format(Column))
        ax.set_xlabel('Number of feature included in iteration')
        ax.set_xticks(X)
        ax.set_xticklabels(AddedColumns,  rotation = 'vertical')
        ax.legend()
        Barplot.savefig('Percentage Plot.png')
        
def TestTreeModelWeights(Feature = 'All'): 
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    if Feature == 'All':
        TestColumns = TestDataSet1.columns
    else: 
        if type(Feature) == str:
            TestColumns = [Feature]
        elif type(Feature) == list:  TestColumns = Feature
        else: print('Feature needs to be of type string or list.')  
    for Column in TestColumns:
        Columns = TestDataSet1.columns
        Columns = Columns.drop(['PRI_nleps','PRI_jets','Events_weight', 'Label'] + [Column])
        TestDataSet = TestDataSet1.drop(Columns,axis=1)
        paramList ={'subsample': 1, 'reg_gamma': 0.4, 'reg_alpha': 0.1, 'n_estimators': 200, 'min_split_loss': 2, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'base_score': 0.9}
        XGBModel = TreeModel(TestDataSet,paramList,SubSampleDataSet=False,ApplyDataCut=False)
        XGBModel.XGBoostTrain()
        AddedColumns = [Column]
        while len(Columns) > 0:
            AddedColumns.append(Columns[0])
            Columns = Columns.drop(Columns[0])
            TestDataSet = TestDataSet1.drop(Columns,axis=1)
            XGBModel = TreeModel(TestDataSet,paramList,SubSampleDataSet=False,ApplyDataCut=False)
            XGBModel.XGBoostTrain()

def SHAPValuesTest(Feature = 'All'): 
    TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
    TestDataSet1 = DataCuts(TestDataSet1)
    TestDataSet1.drop(['EventID'],axis=1,inplace=True)
    if Feature == 'All':
        TestColumns = TestDataSet1.columns
    else: 
        if type(Feature) == str:
            TestColumns = [Feature]
        elif type(Feature) == list:  TestColumns = Feature
        else: print('Feature needs to be of type string or list.')  
    for Column in TestColumns:
        Columns = TestDataSet1.columns
        Columns = Columns.drop(['PRI_nleps','PRI_jets','Events_weight', 'Label'] + [Column])
        TestDataSet = TestDataSet1.drop(Columns,axis=1)
        paramList ={'subsample': 1, 'reg_gamma': 0.4, 'reg_alpha': 0.1, 'n_estimators': 200, 'min_split_loss': 2, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'base_score': 0.9}
        XGBModel = TreeModel(TestDataSet,paramList,SubSampleDataSet=False,ApplyDataCut=False)
        XGBModel.XGBoostTrain()
        XGBModel.SHAPValuePlots()
        AddedColumns = [Column]
        while len(Columns) > 0:
            AddedColumns.append(Columns[0])
            Columns = Columns.drop(Columns[0])
            TestDataSet = TestDataSet1.drop(Columns,axis=1)
            XGBModel = TreeModel(TestDataSet,paramList,SubSampleDataSet=False,ApplyDataCut=False)
            XGBModel.XGBoostTrain()
            XGBModel.SHAPValuePlots()

if "__main__":
   #TestColumns(Feature = 'DER_PT_subleading_ratio_HT',ShowPCAPlots = True)
   #TestColumns(Feature = 'DER_ST_ratio_HT', ShowPCAPlots = False)
   #TestColumns(Feature = 'DER_sum_P_T',ShowPCAPlots = False)
   #TestColumns(Feature = 'PRI_Missing_pt',ShowPCAPlots = False)
   #TestColumns('All')
   #TestTreeModelWeights(Feature = 'DER_PT_subleading_ratio_HT')
   #TestTreeModelWeights(Feature = 'DER_ST_ratio_HT')
   #TestTreeModelWeights(Feature = 'DER_sum_P_T')
   #TestTreeModelWeights(Feature = 'PRI_Missing_pt')
   #SHAPValuesTest(Feature = 'DER_PT_subleading_ratio_HT')
   TestDataSet1 = pd.read_csv(r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\Changing signals\Smuon_400_Neutralino_96\EventData.csv')
   TestDataSet1 = DataCuts(TestDataSet1)
   TestDataSet1.drop(['EventID'],axis=1,inplace=True)
   PCAPlots = PCAPlotter(TestDataSet1,'Label')
   PCAPlots.PCAAnalysis( MinNoofJets= 1, MaxNoofJets=1, MinNoofLeptons = 1, MaxNoofLeptons = 1)
   PCAPlots.PCAAnalysis( MinNoofJets= 1, MaxNoofJets=2, MinNoofLeptons = 1, MaxNoofLeptons = 1)
   PCAPlots.PCAAnalysis( MinNoofJets= 1, MaxNoofJets=1, MinNoofLeptons = 1, MaxNoofLeptons = 2)
   PCAPlots.PCAAnalysis( MinNoofJets= 1, MaxNoofJets=2, MinNoofLeptons = 1, MaxNoofLeptons = 2)



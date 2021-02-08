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

def Pipeline(DataSet, paramList = None,Plot_titles=None):
    DataSet = DataCuts(DataSet)
    
    Key = { 'PRI_nleps' : r'$N_{\ell}$',
           'PRI_jets' : r'$N_{jets}$',
           'PRI_leading_jet_pt' : r'$jet_{PT}^{(1)}$',
           'PRI_subleading_jet_pt' : r'$jet_{PT}^{(2)}$',
           'PRI_leading_jet_eta' : r'$jet_{\eta}^{(1)}$',
           'PRI_subleading_jet_eta' : r'$jet_{\eta}^{(2)}$',
           'PRI_lep_leading_pt' : r'$\ell_{PT}^{(1)}$',
           'PRI_lep_subleading_pt' : r'$\ell_{PT}^{(2)}$',
           'PRI_lep_leading_eta' : r'$\ell_{\eta}^{(1)}$',
           'PRI_lep_subleading_eta' : r'$\ell_{\eta}^{(2)}$',
           'PRI_lep_leading_phi' : r'$\ell_{\phi}^{(1)}$',
           'PRI_lep_subleading_phi' : r'$\ell_{\phi}^{(2)}$',
           'DER_P_T_ratio_lep_pair' : r'$\frac{\ell_{PT}^{(1)}}{\ell_{PT}^{(2)}}$',
           'DER_Diff_Eta_lep_pair' : r'$abs(\ell_{\eta}^{(1)} - \ell_{\eta}^{(2)})$',
           'DER_Diff_Phi_lep_pair' : r'$abs(\ell_{\phi}^{(1)} - \ell_{\phi}^{(2)})$',
           'DER_sum_P_T' : r'$\sum(PT)$',
           'PRI_Missing_pt' : r'MissingPT',
           'DER_PT_leading_lepton_ratio_PT_leading_jet' : r'$\frac{\ell_{PT}^{(1)}}{jet_{PT}^{(1)}}$',
           'DER_PT_leading_lept_ratio_HT' : r'$\frac{\ell_{PT}^{(1)}}{HT}$',
           'DER_ST_ratio_PT_Leading_jet' : r'$\frac{ST}{jet_{PT}^{(1)}}$',
           'DER_ST_ratio_HT' : r'$\frac{ST}{HT}$',
           'DER_PT_subleading_lepton_ratio_PT_leading_jet' : r'$\frac{\ell_{PT}^{(2)}}{jet_{PT}^{(1)}}$',
           'DER_PT_subleading_lepton_ratio_HT' : r'$\frac{\ell_{PT}^{(2)}}{HT}$'   }
    
    try:
        DataSet.drop(['EventID'],axis=1,inplace=True)
    except:
        pass

    PCAPlots = PCAPlotter(DataSet,'Label', Key)
    PCAPlots.PCAAnalysis()
    
   
    
    DataSet.rename(columns=Key,inplace=True) 
    
    if paramList == None:
        XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False) 
        XGBModel.HyperParameterTuning()
    else:
        XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False, paramList=paramList) 
        
    XGBModel.XGBoostTrain()
    MeanSHAPValues = XGBModel.SHAPValuePlots(Plot_titles)
    
    
    MeanPermValues = XGBModel.FeaturePermutation(usePredict_poba=False,Plot_Title=Plot_titles)
    
    #PCAMag = {}
    #for items in PCAPlots.FeaturePCAValues['Leptons 2 Jets 2']:
    #    PCAMag[items] = np.sqrt(sum(abs(PCAPlots.FeaturePCAValues['Leptons 2 Jets 2'][items])))
    #PCAMag.pop('PRI_nleps')
    #PCAMag.pop('PRI_jets')
    #PCAMag = dict(sorted(PCAMag.items(), key=lambda item: item[1]))
    #
    #DropColumns = list(PCAMag.keys())[:8]
    #print(DropColumns) 
    #
    #DataSet.drop(DropColumns,axis=1,inplace=True)
    #DataSet.drop('DER_PT_subleading_lepton_ratio_PT_leading_jet',axis=1,inplace = True)
    
    #PCAPlots = PCAPlotter(DataSet,'Label')
    #PCAPlots.PCAAnalysis()
    
    #if paramList == None:
    #    XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False) 
    #    XGBModel.HyperParameterTuning()
    #else:
    #    XGBModel = TreeModel(DataSet,SubSampleDataSet=False,ApplyDataCut=False, paramList=paramList) 
    #    
    #XGBModel.XGBoostTrain()
    #XGBModel.SHAPValuePlots(Plot_titles)
    return MeanSHAPValues, MeanPermValues
    

def FeatureAxesPlot(Dictionary, ax=None, total_width=0.8, single_width=1,):
        if ax is None:
            ax = plt.gca()
            
        n_bars = len(Dictionary)
        
        
        X = np.arange(len(Dictionary))
        bar_width = total_width / n_bars
        i = 0
        for k in Dictionary.keys():
            
            key = k
            Temp = dict(sorted(Dictionary[k].items()))
            keys = Temp.keys()
            values = Temp.values()
            x_offset = (i - n_bars/2)*bar_width  + bar_width / 2
            
            X = np.arange(len(Temp))
            ax.bar(X + x_offset,values, width=bar_width * single_width ,label=key)
            i += 1
            
        ax.set_xticks(np.arange(len(Dictionary[k])))
        ax.set_xticklabels(list(sorted(Dictionary[k])),rotation='vertical',fontsize=15)
        ax.legend()
        return ax

def FeaturePermutationComparisonPlot(DictCases):
    fig, axes = plt.subplots(nrows = 1, ncols = 3 , figsize=(40 * 3, 40))
    
    for i in range(3):
                 
        if i == 0:
            SMUONMASS = [360, 320]
            NEUTRALINOMASS = [270, 220]
        elif i == 1:
            SMUONMASS = [290, 240, 240, 420, 450, 500, 400]
            NEUTRALINOMASS = [190, 140, 130, 140, 200, 190, 180]
        elif i == 2:
            SMUONMASS= [500, 400, 510, 200, 210, 250]
            NEUTRALINOMASS= [95, 80, 60, 60, 65, 55]
            
        TempDict = dict()
        for k in range(len(SMUONMASS)):
           TempDict['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neatralino_{}'.format(SMUONMASS[k],NEUTRALINOMASS[k])] = DictCases['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neatralino_{}'.format(SMUONMASS[k],NEUTRALINOMASS[k])]
       
        FeatureAxesPlot(TempDict, ax=axes[i])
        axes[i].legend()
       
    axes[0].set_title('Compact case',fontsize = 25)
    axes[1].set_title('Medium compact case',fontsize = 25)
    axes[2].set_title('Split case',fontsize = 25)
    axes[0].set_ylabel('Feature permutation importance', fontsize =  15)
    
    fig.suptitle('Permutation importance')

        
    
def RenameDataBaseColumns(DataSet):
    Key = { 'PRI_nleps' : r'$N_{\ell}$',
           'PRI_jets' : r'$N_{jets}$',
           'PRI_leading_jet_pt' : r'$jet_{PT}^{(1)}$',
           'PRI_subleading_jet_pt' : r'$jet_{PT}^{(2)}$',
           'PRI_leading_jet_eta' : r'$jet_{\eta}^{(1)}$',
           'PRI_subleading_jet_eta' : r'$jet_{\eta}^{(2)}$',
           'PRI_lep_leading_pt' : r'$\ell_{PT}^{(1)}$',
           'PRI_lep_subleading_pt' : r'$\ell_{PT}^{(2)}$',
           'PRI_lep_leading_eta' : r'$\ell_{\eta}^{(1)}$',
           'PRI_lep_subleading_eta' : r'$\ell_{\eta}^{(2)}$',
           'PRI_lep_leading_phi' : r'$\ell_{\phi}^{(1)}$',
           'PRI_lep_subleading_phi' : r'$\ell_{\phi}^{(2)}$',
           'DER_P_T_ratio_lep_pair' : r'$\frac{\ell_{PT}^{(1)}}{\ell_{PT}^{(2)}}$',
           'DER_Diff_Eta_lep_pair' : r'$abs(\ell_{\eta}^{(1)} - \ell_{\eta}^{(2)}$',
           'DER_Diff_Phi_lep_pair' : r'$abs(\ell_{\phi}^{(1)} - \ell_{\phi}^{(2)}$',
           'DER_sum_P_T' : r'$\sum(PT)$',
           'PRI_Missing_pt' : r'MissingPT',
           'DER_PT_leading_lepton_ratio_PT_leading_jet' : r'$\frac{\ell_{PT}^{(1)}}{jet_{PT}^{(1)}}$',
           'DER_PT_leading_lept_ratio_HT' : r'$\frac{\ell_{PT}^{(1)}}{HT}$',
           'DER_ST_ratio_PT_Leading_jet' : r'$\frac{ST}{jet_{PT}^{(1)}}$',
           'DER_ST_ratio_HT' : r'$\frac{ST}{HT}$',
           'DER_PT_subleading_lepton_ratio_PT_leading_jet' : r'$\frac{\ell_{PT}^{(2)}}{jet_{PT}^{(1)}}$',
           'DER_PT_subleading_lepton_ratio_HT' : r'$\frac{\ell_{PT}^{(2)}}{HT}$'   }
    
    DataSet.rename(columns=Key, inplace=True)


def test():
    Smuon_200_Neutranlino_96 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\outputs\Signal Events\Events_PPtoSmuonSmuon_Smuon_Mass_200_Neatralino_96\EventData.csv'
    Smuon_200_Neutralino_195 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\outputs\Signal Events\Events_PPtoSmuonSmuon_Smuon_Mass_200_Neatralino_195\EventData.csv'
    Smuon_400_Neutralino_96 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\outputs\Signal Events\Events_PPtoSmuonSmuon_Smuon_Mass_400_Neatralino_96\EventData.csv'
    Smuon_400_Neutralino_195 = r'I:\Results For Particle Physics\00Gerhard-2020-10-14\DockerOutput_Gerhard\outputs\Signal Events\Events_PPtoSmuonSmuon_Smuon_Mass_400_Neatralino_195\EventData.csv'
    DataSetList = [Smuon_200_Neutranlino_96, Smuon_200_Neutralino_195, Smuon_400_Neutralino_96, Smuon_400_Neutralino_195] 

    NEUTRALINOMASS=[270, 220, 190, 140, 130, 140, 95, 80, 60, 60, 65, 55, 200, 190, 180, 195, 96, 195, 96]
    SMUONMASS=[360, 320, 290, 240, 240, 420, 500, 400, 510, 200, 210, 250, 450, 500, 400, 400, 400, 200, 200]
    
    MeanSHAPValues = dict()
    MeanPermValues = dict()
    
    BackGroundData=pd.read_csv(r'I:\CSV\Background_Events\EventData.csv')
    BackGroundData.drop('EventID',axis=1,inplace=True)    
   
    for i in range(len(NEUTRALINOMASS)):
        Path = 'I:\CSV\Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neatralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i])
        SignalEvents = pd.read_csv(Path)
        
        SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
        DataSet = pd.concat([BackGroundData,SignalEvents])
        DataSet.sample(frac=1) 
        #DataSet.drop(['DER_PT_leading_lepton_ratio_PT_leading_jet', 
        #                   'DER_PT_leading_lept_ratio_HT', 
        #                   'DER_ST_ratio_PT_Leading_jet', 
        #                   'DER_ST_ratio_HT', 
        #                   'DER_PT_subleading_lepton_ratio_PT_leading_jet', 
        #                   'DER_PT_subleading_lepton_ratio_HT'],axis=1,inplace=True)
       
        DataSet.drop(['HT','ST'],axis=1,inplace=True)
        
        paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
        
        MeanSHAPValues['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neatralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])], MeanPermValues['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neatralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = Pipeline(DataSet, paramList = paramList,Plot_titles='Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neatralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
    f = open("I:\Results For Particle Physics\Feature Dictionaries\MeanSHAPValues_No_HT.txt", "w")
    f.write(str(MeanSHAPValues))
    f.close()
    
    f = open("I:\Results For Particle Physics\Feature Dictionaries\MeanPermValues_no_HT.txt", "w")
    f.write(str(MeanPermValues))
    f.close()   
    
    
    return DictCases
    
        #for items in DataSetList:
            #    Pipeline(pd.read_csv(items), paramList = paramList)
    
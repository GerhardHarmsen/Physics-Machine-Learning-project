# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:34:24 2020

@author: gerha
"""

from XGBoosterModel import TreeModel, DataCuts, RemoveFeaturesNotinPaper
from Feature_Plots_PCA import PCAPlotter, FeaturePlots
import Feature_Plots_PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Displotter():
    def __init__(self,DataSet,LabelOfInterest):
        try:
            DataSet[LabelOfInterest]
        except:
           print('Label of interest not found.')
           exit
        try: 
            DataSet['Events_weight']
            print('Event weights detected.')
            self.ContainsWeights = True
        except:
            print('No event weightings detected. Outputs will not consider the event weights.')        
            self.ContainsWeights = False
        self.df = DataSet
        self.lbl = LabelOfInterest
        
        
        
        
    def PairPlotAnalysis(self,MaxNoofJets = 2, MaxNoofLeptons = 2):
        """
        

        Parameters
        ----------
        MaxNoofJets : Int, optional
            DESCRIPTION. The max number of jets to split the plots up into. The default is 2.
        MaxNoofLeptons : int, optional
            DESCRIPTION. The max number of leptons to split the plots up into. The default is 2.
        
        The max number of jet and leptons cannot both be zero as this prevents the plots from being created. Atleast one must be greater than zero.
        Returns
        -------
        None.

        """
        ### Since python starts counting at zero and we are interested in the case of zero jets and leptons we need to increase by one.
        MaxNoofJets = MaxNoofJets + 1
        MaxNoofLeptons = MaxNoofLeptons + 1
        fig, axes = plt.subplots(nrows = MaxNoofJets, ncols = MaxNoofLeptons , figsize=(40, 40))
               
        if (MaxNoofJets == 1) and (MaxNoofLeptons == 1):
           exit('Unable to create reasonable PCA plot. The max number of jets or leptons must be atleast one.')
    
        for Jets in range(MaxNoofJets):
            for Leptons in range(MaxNoofLeptons):
                if Jets == (MaxNoofJets - 1):
                    DispDataSet = self.df[self.df.PRI_jets >= Jets]
                else:
                    DispDataSet = self.df[self.df.PRI_jets == Jets]
            
                if Leptons == (MaxNoofLeptons - 1):
                    DispDataSet = DispDataSet[DispDataSet.PRI_nleps >= Leptons] 
                else:
                    DispDataSet = DispDataSet[DispDataSet.PRI_nleps == Leptons] 
                    
                self.DispPlots(DispDataSet,Jets,Leptons, ax=axes[Jets,Leptons]) 
                try:
                    self.DispPlots(DispDataSet,Jets,Leptons, ax=axes[Jets,Leptons])      
                except:
                    pass
                
                axes[Jets,Leptons].legend()
                PercentageinPlot = len(DispDataSet) / len(self.df) * 100
                axes[Jets,Leptons].set_title('Percentage of the total dataset: {} %'.format(round(PercentageinPlot)),fontsize = 25)
                axes[MaxNoofJets - 1,Leptons].set_xlabel('Number of leptons: {}'.format(Leptons),fontsize = 25)
                axes[Jets,0].set_ylabel('Number of Jets: {}'.format(Jets), fontsize = 25)
            
    
        plt.tight_layout() #This to avoid overlap of labels and titles across plots
        plt.show()
        
    def DispPlots(self,DataSet,NoofJets,NoofLepton,ax=None, plt_kwargs = {}):
        if self.ContainsWeights:
           DataSet2 = DataSet.drop(labels = [self.lbl,'Events_weight'], axis = 1)
        else: 
           DataSet2 = DataSet.drop(labels = self.lbl, axis = 1)
        
        if ax is None:
            ax = plt.gca()
        

        sns.kdeplot(data=DataSet,x='HT',hue='Label',ax=ax,common_norm=False)   
       
       
        return ax


Signal = pd.read_csv(r'I:\Results For Particle Physics\PCA TestsFolder\Signal\Events_PPtoSmuonSmuon_Smuon_Mass_400_Neatralino_96\EventData.csv')
BackGround = pd.read_csv(r'I:\Results For Particle Physics\PCA TestsFolder\Background\Events_PPtoTopTopBar\EventData.csv')

BackGround.Label = 'TTBar'
Signal.Label = 'Signal'

DataSet = pd.concat([BackGround,Signal])

DataSet = DataCuts(DataSet)

AllFeature = RemoveFeaturesNotinPaper(DataSet)

#AllFeature = AllFeature.sample(n = 10000)

FeaturePlots(AllFeature, 'Label')

PairPlots = Displotter(AllFeature,'Label')
PairPlots.PairPlotAnalysis()

sns.displot(AllFeature,x='HT',hue='Label',kind='kde')

PCAPlots = PCAPlotter(AllFeature,'Label')
PCAPlots.PCAAnalysis()

OneFeature = DataSet.drop(['EventID', 'PRI_leading_jet_pt',
       'PRI_subleading_jet_pt', 'PRI_leading_jet_eta',
       'PRI_subleading_jet_eta', 'PRI_lep_leading_pt', 'PRI_lep_subleading_pt',
       'PRI_lep_leading_eta', 'PRI_lep_subleading_eta', 'PRI_lep_leading_phi',
       'PRI_lep_subleading_phi', 'DER_P_T_ratio_lep_pair',
       'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 'DER_sum_P_T',
       'PRI_Missing_pt', 'ST',
       'DER_PT_leading_lepton_ratio_PT_leading_jet',
       'DER_PT_leading_lept_ratio_HT', 'DER_ST_ratio_PT_Leading_jet',
       'DER_ST_ratio_HT', 'DER_PT_subleading_lepton_ratio_PT_leading_jet',
       'DER_PT_subleading_lepton_ratio_HT'],axis=1)

OneFeature = OneFeature.sample(n = 10000)

FeaturePlots(OneFeature, 'Label')

PairPlots = Displotter(OneFeature,'Label')
PairPlots.PairPlotAnalysis()

sns.displot(OneFeature,x='HT',hue='Label',kind='kde')

PCAPlots = PCAPlotter(OneFeature,'Label')
PCAPlots.PCAAnalysis()

#XGBModel = TreeModel(OneFeature, ApplyDataCut = False)
#XGBModel.HyperParameterTuning()
#XGBModel.XGBoostTrain()


TwoFeatures = DataSet.drop(['EventID', 'PRI_leading_jet_pt',
       'PRI_subleading_jet_pt', 'PRI_leading_jet_eta',
       'PRI_subleading_jet_eta', 'PRI_lep_leading_pt', 'PRI_lep_subleading_pt',
       'PRI_lep_leading_eta', 'PRI_lep_subleading_eta', 'PRI_lep_leading_phi',
       'PRI_lep_subleading_phi', 'DER_P_T_ratio_lep_pair',
       'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 'DER_sum_P_T',
       'PRI_Missing_pt',
       'DER_PT_leading_lepton_ratio_PT_leading_jet',
       'DER_PT_leading_lept_ratio_HT', 'DER_ST_ratio_PT_Leading_jet',
       'DER_ST_ratio_HT', 'DER_PT_subleading_lepton_ratio_PT_leading_jet',
       'DER_PT_subleading_lepton_ratio_HT'],axis=1)

TwoFeatures = TwoFeatures.sample(n = 10000)

PairPlots = Displotter(TwoFeatures,'Label')
PairPlots.PairPlotAnalysis()
for column in TwoFeatures.drop('Label',axis=1).columns:
    sns.displot(TwoFeatures,x=column,hue='Label',kind='kde')

FeaturePlots(TwoFeatures, 'Label')

PCAPlots = PCAPlotter(TwoFeatures,'Label')
PCAPlots.PCAAnalysis()

#XGBModel = TreeModel(TwoFeatures, ApplyDataCut = False)
#XGBModel.HyperParameterTuning()
#XGBModel.XGBoostTrain()

ThreeFeatures = DataSet.drop(['EventID', 'PRI_leading_jet_pt',
       'PRI_subleading_jet_pt', 'PRI_leading_jet_eta',
       'PRI_subleading_jet_eta', 'PRI_lep_leading_pt', 'PRI_lep_subleading_pt',
       'PRI_lep_leading_eta', 'PRI_lep_subleading_eta', 'PRI_lep_leading_phi',
       'PRI_lep_subleading_phi', 'DER_P_T_ratio_lep_pair',
       'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 'DER_sum_P_T',
       'DER_PT_leading_lepton_ratio_PT_leading_jet',
       'DER_PT_leading_lept_ratio_HT', 'DER_ST_ratio_PT_Leading_jet',
       'DER_ST_ratio_HT', 'DER_PT_subleading_lepton_ratio_PT_leading_jet',
       'DER_PT_subleading_lepton_ratio_HT'],axis=1)

ThreeFeatures = ThreeFeatures.sample(n = 10000)

PairPlots = Displotter(TwoFeatures,'Label')
PairPlots.PairPlotAnalysis()
for column in ThreeFeatures.drop('Label',axis=1).columns:
    sns.displot(ThreeFeatures,x=column,hue='Label',kind='kde')

PCAPlots = PCAPlotter(ThreeFeatures,'Label')
PCAPlots.PCAAnalysis()

FeaturePlots(ThreeFeatures, 'Label')

#XGBModel = TreeModel(ThreeFeatures, ApplyDataCut = False)
#XGBModel.HyperParameterTuning()
#XGBModel.XGBoostTrain()


########################################################################
### BackGround Combinations
########################################################################

BackGround = pd.read_csv(r'I:\Results For Particle Physics\PCA TestsFolder\Background\Events_PP_WW_lvl\EventData.csv')

BackGround.Label = 'WWBackGround'

DataSet = pd.concat([BackGround,DataSet])

DataSet = DataCuts(DataSet)

OneFeature = DataSet.drop(['EventID', 'PRI_leading_jet_pt',
       'PRI_subleading_jet_pt', 'PRI_leading_jet_eta',
       'PRI_subleading_jet_eta', 'PRI_lep_leading_pt', 'PRI_lep_subleading_pt',
       'PRI_lep_leading_eta', 'PRI_lep_subleading_eta', 'PRI_lep_leading_phi',
       'PRI_lep_subleading_phi', 'DER_P_T_ratio_lep_pair',
       'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 'DER_sum_P_T',
       'PRI_Missing_pt', 'ST',
       'DER_PT_leading_lepton_ratio_PT_leading_jet',
       'DER_PT_leading_lept_ratio_HT', 'DER_ST_ratio_PT_Leading_jet',
       'DER_ST_ratio_HT', 'DER_PT_subleading_lepton_ratio_PT_leading_jet',
       'DER_PT_subleading_lepton_ratio_HT'],axis=1)

OneFeature = OneFeature.sample(n = 10000)

FeaturePlots(OneFeature, 'Label')

PairPlots = Displotter(OneFeature,'Label')
PairPlots.PairPlotAnalysis()

sns.displot(OneFeature,x='HT',hue='Label',kind='kde')

PCAPlots = PCAPlotter(OneFeature,'Label')
PCAPlots.PCAAnalysis()

#XGBModel = TreeModel(OneFeature, ApplyDataCut = False)
#XGBModel.HyperParameterTuning()
#XGBModel.XGBoostTrain()


TwoFeatures = DataSet.drop(['EventID', 'PRI_leading_jet_pt',
       'PRI_subleading_jet_pt', 'PRI_leading_jet_eta',
       'PRI_subleading_jet_eta', 'PRI_lep_leading_pt', 'PRI_lep_subleading_pt',
       'PRI_lep_leading_eta', 'PRI_lep_subleading_eta', 'PRI_lep_leading_phi',
       'PRI_lep_subleading_phi', 'DER_P_T_ratio_lep_pair',
       'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 'DER_sum_P_T',
       'PRI_Missing_pt',
       'DER_PT_leading_lepton_ratio_PT_leading_jet',
       'DER_PT_leading_lept_ratio_HT', 'DER_ST_ratio_PT_Leading_jet',
       'DER_ST_ratio_HT', 'DER_PT_subleading_lepton_ratio_PT_leading_jet',
       'DER_PT_subleading_lepton_ratio_HT'],axis=1)

#TwoFeatures = TwoFeatures.sample(n = 10000)

PairPlots = Displotter(TwoFeatures,'Label')
PairPlots.PairPlotAnalysis()
for column in TwoFeatures.drop('Label',axis=1).columns:
    sns.displot(TwoFeatures,x=column,hue='Label',kind='kde')

FeaturePlots(TwoFeatures, 'Label')

PCAPlots = PCAPlotter(TwoFeatures,'Label')
PCAPlots.PCAAnalysis()

#XGBModel = TreeModel(TwoFeatures, ApplyDataCut = False)
#XGBModel.HyperParameterTuning()
#XGBModel.XGBoostTrain()

ThreeFeatures = DataSet.drop(['EventID', 'PRI_leading_jet_pt',
       'PRI_subleading_jet_pt', 'PRI_leading_jet_eta',
       'PRI_subleading_jet_eta', 'PRI_lep_leading_pt', 'PRI_lep_subleading_pt',
       'PRI_lep_leading_eta', 'PRI_lep_subleading_eta', 'PRI_lep_leading_phi',
       'PRI_lep_subleading_phi', 'DER_P_T_ratio_lep_pair',
       'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 'DER_sum_P_T',
       'DER_PT_leading_lepton_ratio_PT_leading_jet',
       'DER_PT_leading_lept_ratio_HT', 'DER_ST_ratio_PT_Leading_jet',
       'DER_ST_ratio_HT', 'DER_PT_subleading_lepton_ratio_PT_leading_jet',
       'DER_PT_subleading_lepton_ratio_HT'],axis=1)

ThreeFeatures = ThreeFeatures.sample(n = 10000)

PairPlots = Displotter(TwoFeatures,'Label')
PairPlots.PairPlotAnalysis()
for column in ThreeFeatures.drop('Label',axis=1).columns:
    sns.displot(ThreeFeatures,x=column,hue='Label',kind='kde')

PCAPlots = PCAPlotter(ThreeFeatures,'Label')
PCAPlots.PCAAnalysis()

FeaturePlots(ThreeFeatures, 'Label')

#XGBModel = TreeModel(ThreeFeatures, ApplyDataCut = False)
#XGBModel.HyperParameterTuning()
#XGBModel.XGBoostTrain()
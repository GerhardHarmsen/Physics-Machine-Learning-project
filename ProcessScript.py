# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:58:43 2021

@author: gerha
"""

from XGBoosterModel import TreeModel, DataCuts, xgb_f1
from Feature_Plots_PCA import PCAPlotter
import Feature_Plots_PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
import sys

NoofCPU = cpu_count()
sys.path.insert(0, r'I:\Google Drive\Research\Post Doc\Python Codes')

PlotTitleSize = 80
PlotLabelSize = 60

##### These are the cut offs for the diffirent mass regimes. The valuess here refer to the ratio Smuon/Neutralino

CompressedCutOff = 1.2
NeutralCutOff = 3

#### Known mass cases

#### SMuon Neutralino mass cases
NEUTRALINOMASS=[270, 220, 190, 140, 130, 140, 95, 80, 60, 60, 65, 55, 200, 190, 180, 195, 96, 195, 96, 175, 87, 125, 100, 70, 100, 68, 120, 150, 75, 300, 500, 440, 260]
SMUONMASS=[360, 320, 290, 240, 240, 420, 500, 400, 510, 200, 210, 250, 450, 500, 400, 400, 400, 200, 200, 350, 350, 375, 260, 350, 300, 275, 475, 300, 450, 310, 510, 450, 275]


#### Locations of the  CSV Files########
CSVLOCATION = r'G:\CSV_Output_2021-06-25\CSV'
BACKGROUNDEVENTPATH = r'Background_Events\EventData.csv'
TESTBACKGROUNDEVENTPATH = r'Background_Test_Events\EventData.csv'
HYPERPARAMETERLOCATION = r'G:\CSV\HyperparameterDictionary.json'

### Begin function declaration ######
def Test_for_Files():
    StopRun=False

    for i in range(len(SMUONMASS)):
        if os.path.exists(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i]))) == False:
            print('Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{} is missing.'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
            StopRun=True
            
    if StopRun:
        print('Signal files missing stopping the run')
        sys.exit()
    
    if os.path.exists(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH)) == False:
        print('Background event datset is missing.')
        sys.exit()
    
    if os.path.exists(os.path.join(CSVLOCATION,TESTBACKGROUNDEVENTPATH)) == False:
        print('Background test event datset is missing.')
        sys.exit()
    
    if os.path.exists(os.path.join(HYPERPARAMETERLOCATION)) == False:
        print('Hyperparameter JSON file not found.')
        if input("Do you wwant to continue? y/n") == "n":
            sys.exit()
        

def GetHyperParameters(Smuon,Neutralino):
    JSONParameters = RetrieveDictionary(HYPERPARAMETERLOCATION)
    try:
        paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(Smuon,Neutralino)]
    except:
        MinDist = 999
        for i in range(len(SMUONMASS)):
            dist = np.sqrt((Smuon - SMUONMASS[i])**2 + (Neutralino - NEUTRALINOMASS[i])**2)
            if dist < MinDist and dist > 0:
                try: 
                    paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])]
                    MinDist = dist
                except:
                    print()
   
    return paramList

def SelectLabels_of_interest(Dict,No_of_labels):
    """
    This function selects the labels that are most useful according to the feature permuatation analysis.

    Parameters
    ----------
    Dict : dict.
        Dictionary containing the labels and permutation importances.
    No_of_labels : TYPE
        Numbor of labels to consider after removing those that do not contribute to the model.

    Returns
    -------
    TYPE
        List of labels to be used in the new model.

    """
    List_of_labels = list()
    Dict_of_Labels_and_values = dict()
    for k in Dict.keys():
        Temp = dict(sorted(Dict[k].items(),key=lambda item: item[1],reverse=True))
        Temp = dict(list(Temp.items())[:No_of_labels])
        for keys in Temp.keys():
            if keys not in Dict_of_Labels_and_values.keys():
                Dict_of_Labels_and_values[keys] = Temp[keys]
            elif Dict_of_Labels_and_values[keys] < Temp[keys]:
                Dict_of_Labels_and_values[keys] = Temp[keys]

    Dict_of_Labels_and_values = dict(sorted(Dict_of_Labels_and_values.items(),key=lambda item: item[1],reverse=True))
    List_of_labels = dict(list(Dict_of_Labels_and_values.items())[:No_of_labels])
    print(List_of_labels)
    return list(List_of_labels.keys())
    
def Convert_to_Label(String, Style = 1):
    Neutralino = int(String[(String.rindex('_')+1):])
    Smuon = int(String[(String.rindex('Mass_')+5):(String.rindex('Mass_')+8)])
    if Style == 1 :
        Smuon_string = r'$m_{\bar{\mu}}$'
        Neutralino_string = r'$m_{\bar{\chi}}$'
        String = '{} = {} GeV, {} = {} GeV'.format(Smuon_string, Smuon, Neutralino_string, Neutralino)
    elif Style == 2:
        String = '({}, {})'.format(Smuon, Neutralino)
    return String
        

def FeatureAxesPlot(Dictionary, ax=None, total_width=0.8, single_width=1,YAxisTicks=None,Max_No_of_Labels=None):
        """
        This is an internal function used by FeaturePermutationComparisonPlot
    
        Parameters
        ----------
        Dictionary : Dictionary
            This must be a dictionary of the type generated by the below comparison functions {Mass Case: {Comparison : Value}}.
        ax : TYPE, Axis
            Plot axis to be used for plotting. The default is None.
        total_width : float32, optional
            Largest width of the columns. The default is 0.8.
        single_width : float, optional
            Single column width. The default is 1.
        YAxisTicks : List with YAxis tick values, optional
            Numpy array of YAxis ticks for customising the Y Axis. The default is None.
        Max_No_of_Labels : Integer, optional
            Maximum number of labels to add to the X Axis. The default is None.

        Returns
        -------
        ax : Axis
            Returns the plot on the specified axis.

        """
        if ax is None:
            ax = plt.gca()
            
        n_bars = len(Dictionary)
        
        
        X = np.arange(len(Dictionary))
        bar_width = total_width / n_bars
        i = 0
        YMax = 0
        New_dict = dict()
        
        if Max_No_of_Labels is not None:
            ListofLabels = SelectLabels_of_interest(Dictionary,Max_No_of_Labels)
            for keys in Dictionary.keys():
                New_dict[keys] = { Keys : Dictionary[keys][Keys] for Keys in ListofLabels}
                
            Dictionary = New_dict
            
        else:
            Max_No_of_Labels = len(Dictionary[list(Dictionary.keys())[0]])
            ListofLabels = SelectLabels_of_interest(Dictionary,Max_No_of_Labels)
        
          
        for k in Dictionary.keys():
            key = k
            #Temp = dict(sorted(Dictionary[k].items()))
            #keys = Temp.keys()
            #values = Temp.values()
            keys = Dictionary[k].keys()
            values = Dictionary[k].values()
           
            x_offset = (i - n_bars/2)*bar_width  + bar_width / 2
            
            #X = np.arange(len(Temp))
            #if max(Temp.values()) > YMax:
            #    YMax = max(Temp.values())
            
            X = np.arange(len(Dictionary[k]))
            if max(Dictionary[k].values()) > YMax:
                YMax = max(Dictionary[k].values())
            
            try:
              from shap.plots import colors
              from shap.plots import _utils
              color = colors.red_blue
              color = _utils.convert_color(color)
              mycols=["darkorange", "cornflowerblue", "tab:olive", "teal", "orangered", "gold", "lightpink", "mediumaquamarine", "red", "deepskyblue", "lightcoral", "powderblue", "yellowgreen"]
              ax.bar(X + x_offset,values, width=bar_width * single_width ,label=Convert_to_Label(key,2), color = mycols[i])
              
            except:
              ax.bar(X + x_offset,values, width=bar_width * single_width ,label=Convert_to_Label(key,2))
            i += 1
         
        ax.set_xticks(np.arange(len(Dictionary[k])))
        
        if type(YAxisTicks) is np.ndarray:
            ax.set_yticks(YAxisTicks) 
        
        #ax.set_xticklabels(ListofLabels,rotation='vertical',fontsize=PlotLabelSize)
        ax.set_xticklabels(list(Dictionary[k]),rotation='vertical',fontsize=PlotLabelSize)
        ax.tick_params(axis='y', labelsize=PlotLabelSize)
        ax.legend(fontsize=PlotLabelSize)
        return ax

def FeaturePermutationComparisonPlot(DictCases, PlotTitle='', YLabel ='', YAxisTicks = None, Max_No_of_Labels=None):
    """
    This will generate the plots for the three mass case types "neutral", "compact" and "seperated".   
    The dictionary must be of the form {"Smuon_Mass_{}_Nuetralino_{}" : {"Test" : Value, ...}, ...}, this will let the function split the mass cases in the three mass types. 
    The "Test" values will be used for the x-axis ticks and the Value will be used to plot the bar graphs. Value can be of type float.
     
    
    Parameters
    ----------
    DictCases : Dictionary
        Dictionary containing values to be plotted must of form {"Smuon_Mass_{}_Nuetralino_{}" : {"Test" : Value, ...}, ...}.
    PlotTitle : String, optional
        Title for the plot to be generated, note this is the plot title for the whole plot and not each individual plot. The default is ''.
    YLabel : String, optional
        Will provide the label for the Y axis. The default is ''.
    YAxisTicks : List, optional
        Numpy array of values to put on the y axis. The default is None.
    Max_No_of_Labels : Integer, optional
        Maximum number of ticks on the X-Axis. This will show only the labels with the largest values accosiated to them for all the mass cases. The default is None.

    Returns
    -------
    Plot of the mass comparisons for the three types of mass cases.

    """
    fig, axes = plt.subplots(nrows = 1, ncols = 3 , figsize=(40 * 3, 40))
    if Max_No_of_Labels is not None:
        try: 
            val = int(Max_No_of_Labels)
        except ValueError:
            print("Max_No_of_Labels must be of type int")
    
    CompactCase = list()
    MediumCase = list()
    SeperatedCases = list()
    InnerKeys = list(DictCases.keys())   
    for items in InnerKeys:
        Neutralino = int(items[(items.rindex('_')+1):])
        Smuon = int(items[(items.rindex('Mass_')+5):(items.rindex('Mass_')+8)])
        if Smuon/Neutralino <=1.2:
            CompactCase.append([Smuon,Neutralino])
        elif Smuon/Neutralino > 1.2 and Smuon/Neutralino <= 3:
            MediumCase.append([Smuon,Neutralino])
        else:
            SeperatedCases.append([Smuon,Neutralino])
            
    for i in range(3):
                 
        if i == 0:
            SMUONMASS = [k[0] for k in CompactCase]
            NEUTRALINOMASS = [k[1] for k in CompactCase]
        elif i == 1:
            SMUONMASS = [k[0] for k in MediumCase]
            NEUTRALINOMASS = [k[1] for k in MediumCase]
        elif i == 2:
            SMUONMASS = [k[0] for k in SeperatedCases]
            NEUTRALINOMASS = [k[1] for k in SeperatedCases]
            
        TempDict = dict()
        for k in range(len(SMUONMASS)):
            try:
                TempDict['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[k],NEUTRALINOMASS[k])] = DictCases['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[k],NEUTRALINOMASS[k])]
            except:
                TempDict['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[k],NEUTRALINOMASS[k])] = DictCases['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[k],NEUTRALINOMASS[k])]
        try:
            FeatureAxesPlot(TempDict, ax=axes[i], YAxisTicks=YAxisTicks,Max_No_of_Labels=Max_No_of_Labels)
            #axes[i].legend(fontsize=PlotLabelSize)
            axes[i].legend(fontsize=PlotLabelSize,mode='expand',ncol=3)
        except ZeroDivisionError:
            print('Dictionary of length zero given to plotting function.')
    axes[0].set_title('Compact case',fontsize = PlotTitleSize)
    axes[1].set_title('Neutral case',fontsize = PlotTitleSize)
    axes[2].set_title('Seperated case',fontsize = PlotTitleSize)
    axes[0].set_ylabel(YLabel, fontsize =  PlotLabelSize)
    
    fig.suptitle(PlotTitle,fontsize=PlotTitleSize)
    plt.show()

        
    
def RenameDataBaseColumns(DataSet):
    """
    Function for converting the column headings in the database to Latex style labels for easier reading in the plots

    Parameters
    ----------
    DataSet : Pandas database
        Database to be converted.

    Returns
    -------
    Is done in place.

    """
    Key = { 'PRI_nleps' : r'$n_{\ell}$',
           'PRI_jets' : r'$n_{j}$',
           'PRI_leading_jet_pt' : r'$p_T^{j_1}$', # $jet_{PT}^{(1)}$',
           'PRI_subleading_jet_pt' : r'$p_T^{j_2}$', # $jet_{PT}^{(2)}$',
           'PRI_leading_jet_eta' : r'$\eta^{j_1}$', # $jet_{\eta}^{(1)}$',
           'PRI_subleading_jet_eta' : r'$\eta^{j_2}$', # $jet_{\eta}^{(2)}$',
           'PRI_lep_leading_pt' : r'$p_T^{\ell_1}$', # $\ell_{PT}^{(1)}$',
           'PRI_lep_subleading_pt' : r'$p_T^{\ell_2}$', # $\ell_{PT}^{(2)}$',
           'PRI_lep_leading_eta' : r'$\eta^{\ell_1}$', # $\ell_{\eta}^{(1)}$',
           'PRI_lep_subleading_eta' : r'$\eta^{\ell_2}$', # $\ell_{\eta}^{(2)}$',
           'PRI_lep_leading_phi' : r'$\phi^{\ell_1}$', # $\ell_{\phi}^{(1)}$',
           'PRI_lep_subleading_phi' : r'$\phi^{\ell_2}$', # $\ell_{\phi}^{(2)}$',
           'DER_P_T_ratio_lep_pair' : r'$p_T^{\ell_1}/p_T^{\ell_2}$', # $\frac{\ell_{PT}^{(1)}}{\ell_{PT}^{(2)}}$',
           'DER_Diff_Eta_lep_pair' : r'$|\Delta \eta(\ell_1,\ell_2)|$', # $abs(\ell_{\eta}^{(1)} - \ell_{\eta}^{(2)})$',
           'DER_Diff_Phi_lep_pair' : r'$|\Delta \phi(\ell_1,\ell_2)|$', # $abs(\ell_{\phi}^{(1)} - \ell_{\phi}^{(2)})$',
           'DER_sum_P_T' : r'$\Sigma(p_T)$',
           'PRI_Missing_pt' : r'$Miss.~p_T$',
           'HT' : r'$H_T$',
           'ST' : r'$S_T$',
           'DER_PT_leading_lepton_ratio_PT_leading_jet' : r'$p_T^{\ell_1}/p_T^{j_1}$', # $\frac{\ell_{PT}^{(1)}}{jet_{PT}^{(1)}}$',
           'DER_PT_leading_lept_ratio_HT' : r'$p_T^{\ell_1}/H_T$', # $\frac{\ell_{PT}^{(1)}}{HT}$',
           'DER_ST_ratio_PT_Leading_jet' : r'$S_T/p_T^{j_1}$', # $\frac{ST}{jet_{PT}^{(1)}}$',
           'DER_ST_ratio_HT' : r'$S_T/H_T$', # $\frac{ST}{HT}$',
           'DER_PT_subleading_lepton_ratio_PT_leading_jet' : r'$p_T^{\ell_2}/p_T^{j_1}$', # $\frac{\ell_{PT}^{(2)}}{jet_{PT}^{(1)}}$',
           'DER_PT_subleading_lepton_ratio_HT' : r'$p_T^{\ell_2}/H_T$',  # $\frac{\ell_{PT}^{(2)}}{HT}$’ }
           'PRI_nMuons' : r'$n_{\mu}$', #Number of Muons
           'PRI_Muon_leading_pt' : r'$p_T^{\mu_1}$', 
           'PRI_Muon_subleading_pt' : r'$p_T^{\mu_2}$',
           'PRI_Muon_leading_eta' : r'$\eta^{\mu_1}$',
           'PRI_Muon_subleading_eta' : r'$\eta^{\mu_2}$',
           'PRI_Muon_leading_phi' : r'$\phi^{\mu_1}$',
           'PRI_Muon_subleading_phi' : r'$\phi^{\mu_2}$',
           'DER_Muon_invariant_mass' : r'$M_{T}^{\mu}$',
           'DER_MT2_variable' : r'$M_{T^{2}}$'}
    
    DataSet.rename(columns=Key, inplace=True)
    
def SaveDictionary(FileName,DictionaryToSave):
    """
    Saves a JSON dictionary in the location FileName.

    Parameters
    ----------
    FileName : String
        Location and name of the file to be saved.
    DictionaryToSave : Dictionary
        The dictionary that should be saved.

    Returns
    -------
    None.

    """
    head, tail = os.path.split(FileName)
    if os.path.exists(head) == False:
        os.makedirs(head)
    
    with open(FileName, "w") as myFile:
        json.dump(DictionaryToSave, myFile)
        
def RetrieveDictionary(FileLocation):
    """
    Retrieves a JSON file to be stored as a dictionary.

    Parameters
    ----------
    FileLocation : Sring
        Path to the file.

    Returns
    -------
    Dictionary
        Dictionary converted from the JSON file.

    """
    with open(FileLocation, "r") as myFile:
        return json.load(myFile)

def runTrainingAMSScore(SMuon_Neutralino):
    SMuon, Neutralino, UseF1Score, ResultsLocation = SMuon_Neutralino
    DictReturn = dict()
    BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
    BackGroundData.drop('EventID',axis=1,inplace=True)    
    BackGroundDataTest=pd.read_csv(os.path.join(CSVLOCATION,TESTBACKGROUNDEVENTPATH))
    BackGroundDataTest.drop('EventID',axis=1,inplace=True)   
        
    SignalEvents = pd.read_csv(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMuon,Neutralino)))
    SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
    DataSet = pd.concat([BackGroundData,SignalEvents])
    DataSet.sample(frac=1)
            
    DataSet = DataCuts(DataSet)
            
    RenameDataBaseColumns(DataSet)
            
    JSONParameters = RetrieveDictionary(HYPERPARAMETERLOCATION)
    try:
        paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)]
    except:
        paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}

    paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
    
        
    XGBModel = TreeModel(DataSet,paramList = paramList,ApplyDataCut=False) 
            
    XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    
    AMSTrainScore = XGBModel.AMSScore(XGBModel.TrainingData,0.5)
    AMSTestingScore = XGBModel.AMSScore(XGBModel.TestingData,0.5)
      
    
    DictReturn['Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)] = { 'AMS Score Training' : AMSTrainScore,
                                                                           'AMS Score Testing ' :AMSTestingScore}
       

    FileName = 'Smuon_Mass_{}_Neutralino_{}_Scores.json'.format(SMuon,Neutralino)
    Path = os.path.join(ResultsLocation,'AMS_Train_Test_Score')
    SaveDictionary(os.path.join(Path,FileName),DictReturn)   
    
    
    return 'Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)
    


def runComparison(SMuon_Neutralino, paramList = None):
    """
    Trains the model on the provided database and then compares the model to the other mass cases to determine the generalisablity of the model.
    Saves the results of the comparison in a JSON file called Smuon_Mass_{}_Neutralino_{}_Scores.format(Smuon,Neutralino)
    
    Parameters
    ----------
    SMuon_Neutralino : Tuple of the form [Smuon mass, Neutralino mass]
        Smuon Neutralino mass case that we want to train on.

    Returns
    -------
    String
        String of the name of the saved dictionary.

    """
    SMuon, Neutralino, UseF1Score, ResultsLocation = SMuon_Neutralino
    DictReturn = dict()
    BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
    BackGroundData.drop('EventID',axis=1,inplace=True)    
    BackGroundDataTest=pd.read_csv(os.path.join(CSVLOCATION,TESTBACKGROUNDEVENTPATH))
    BackGroundDataTest.drop('EventID',axis=1,inplace=True)   
        
    SignalEvents = pd.read_csv(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMuon,Neutralino)))
    SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
    DataSet = pd.concat([BackGroundData,SignalEvents])
    DataSet.sample(frac=1)
            
    DataSet = DataCuts(DataSet)
            
    RenameDataBaseColumns(DataSet)
            
    if paramList == None:
        paramList = GetHyperParameters(SMuon,Neutralino)
        #try:
        #    paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)]
        #except:
        #    ApproxMass = AproxHyperParameters(SMuon,Neutralino)
        #    paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(ApproxMass[0],ApproxMass[1])]


    XGBModel = TreeModel(DataSet,paramList = paramList,ApplyDataCut=False) 
            
    XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    
    #XGBModel.XGBoostTrain(['ams@0.15'])
    
    SelfTestScore = XGBModel.AMSScore(XGBModel.TestingData,Threshold=0.5)
      
    Results = dict()
        
    for k in range(len(SMUONMASS)):
            SignalEvents = pd.read_csv(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[k],NEUTRALINOMASS[k])))
            SignalEvents.drop(['EventID'],axis=1,inplace=True)  
            
            DataSet = pd.concat([BackGroundDataTest,SignalEvents])
            DataSet.sample(frac=1)
    
            DataSet = DataCuts(DataSet)
    
            RenameDataBaseColumns(DataSet)
            
            F1Score = XGBModel.ModelPredictions(DataSet, Metric='f1')
            AUCScores = XGBModel.ModelPredictions(DataSet, Metric='auc')
            SigWeight = DataSet.Events_weight[DataSet.Label == 1].sum()
            Threshold=0.5
            if SMUONMASS[k] == SMuon:
                Score = SelfTestScore
            else:
                Score = XGBModel.AMSScore(DataSet,Threshold=Threshold)

            
            Results['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[k],NEUTRALINOMASS[k])] = {
                                                                                'AMS Score' :Score,
                                                                                'F1 Score' : F1Score,
                                                                                'auc Score' : AUCScores,
                                                                                'Signal Weight' : SigWeight}
    DictReturn['Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)]=Results
    FileName = 'Smuon_Mass_{}_Neutralino_{}_Scores.json'.format(SMuon,Neutralino)
    Path = os.path.join(ResultsLocation,'Exclusion_plot_results')
    SaveDictionary(os.path.join(Path,FileName),DictReturn)  
    return 'Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)

def SHAP_Perm_Test(ResultsLocation):
    
    MeanSHAPValues = dict()
    MeanPermValues = dict()
    
    BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
    BackGroundData.drop('EventID',axis=1,inplace=True)    
   
    for i in range(len(NEUTRALINOMASS)):
        Path = os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
        SignalEvents = pd.read_csv(Path)
        
        SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
        DataSet = pd.concat([BackGroundData,SignalEvents])
        DataSet.sample(frac=1)
        
        paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
        
        MeanSHAPValues['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])], MeanPermValues['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = Pipeline(DataSet, paramList = paramList,Plot_titles='Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
    
    
    SaveDictionary(os.path.join(ResultsLocation,'Feature Dictionaries\MeanSHAPValues_No_ratios.txt'),MeanSHAPValues)
    
    SaveDictionary(os.path.join(ResultsLocation,'Feature Dictionaries\MeanPermValues_No_ratios.txt'),MeanPermValues)

def CompareModelwithandwithoutratios(DataSet,UseF1Score=False):
    #### Train model
    Threshold = 0.9
    JSONParameters = RetrieveDictionary(HYPERPARAMETERLOCATION)
    try:
        paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)]
    except:
        paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
        
    
    DataSet = DataCuts(DataSet)
          
    XGBModel = TreeModel(DataSet,ApplyDataCut=False,  paramList=paramList) 
    
    XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    #if UseF1Score:
    #    PermList = XGBModel.FeaturePermutation(usePredict_poba=False,scoreFunction='F1',Plot_Title='Test')
    #else:
    #    PermList = XGBModel.FeaturePermutation(usePredict_poba=False,scoreFunction='AUC',Plot_Title='Test')

    #RemoveList = [ PermList[i][-1] for i in range(len(PermList)) ]
    #NewDataSet = DataSet.copy()
    #NewDataSet.drop(RemoveList[9:],axis=1,inplace=True)
    #XGBModel = TreeModel(NewDataSet,paramList = paramList,ApplyDataCut=False)
    #XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    
    AMSScore = dict()
    Score = XGBModel.AMSScore(DataSet,Threshold=Threshold)
    while Score == 0:
        Threshold = Threshold - 0.1
        print(Threshold)
        Score = XGBModel.AMSScore(DataSet,Threshold=Threshold)
    
    AMSScore['All_features'] = Score
    
    ### No HT
    
    DataSet2 = DataSet.drop(['HT','ST'],axis=1)
    
    XGBModel = TreeModel(DataSet2,ApplyDataCut=False,  paramList=paramList) 
    
    XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    
    #if UseF1Score:
    #    PermList = XGBModel.FeaturePermutation(usePredict_poba=False,scoreFunction='F1',Plot_Title='Test')
    #else:
    #    PermList = XGBModel.FeaturePermutation(usePredict_poba=False,scoreFunction='AUC',Plot_Title='Test')

    #RemoveList = [ PermList[i][-1] for i in range(len(PermList)) ]
    #NewDataSet = DataSet2.copy()
    #if len(NewDataSet.columns) > 9:
    #    NewDataSet.drop(RemoveList[9:],axis=1,inplace=True)
    
    #XGBModel = TreeModel(NewDataSet,paramList = paramList,ApplyDataCut=False)
    #XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    
    
    Score = XGBModel.AMSScore(DataSet2,Threshold=Threshold)
    while Score == 0:
        Threshold = Threshold - 0.1
        Score =  XGBModel.AMSScore(DataSet2,Threshold=Threshold)
    
    AMSScore['NO_HT'] = Score
    
    ### Noratios
     
    DataSet2 = DataSet.drop(['DER_PT_leading_lepton_ratio_PT_leading_jet', 
                             'DER_PT_leading_lept_ratio_HT', 
                             'DER_ST_ratio_PT_Leading_jet', 
                             'DER_ST_ratio_HT', 
                             'DER_PT_subleading_lepton_ratio_PT_leading_jet', 
                             'DER_PT_subleading_lepton_ratio_HT'],axis=1)
    
    XGBModel = TreeModel(DataSet2,ApplyDataCut=False,  paramList=paramList) 
    
    XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    #if UseF1Score:
    #    PermList = XGBModel.FeaturePermutation(usePredict_poba=False,scoreFunction='F1',Plot_Title='Test')
    #else:
    #    PermList = XGBModel.FeaturePermutation(usePredict_poba=False,scoreFunction='AUC',Plot_Title='Test')

    #RemoveList = [ PermList[i][-1] for i in range(len(PermList)) ]
    #NewDataSet = DataSet2.copy()
    #if len(NewDataSet.columns) > 9:
    #    NewDataSet.drop(RemoveList[9:],axis=1,inplace=True)
    
    #XGBModel = TreeModel(NewDataSet,paramList = paramList,ApplyDataCut=False)
    #XGBModel.XGBoostTrain(UseF1Score=UseF1Score)
    
    Score = XGBModel.AMSScore(DataSet2,Threshold=Threshold)
    while Score == 0: 
        Threshold = Threshold - 0.1
        Score = XGBModel.AMSScore(DataSet2,Threshold=Threshold)
    
    AMSScore['NO_ratio'] = Score
    
    
    return AMSScore

########### Functions called by the main execusion
def PCA_Plotter():
     AllDataSets = input("Do you want to look at the PCA plots for all of the mass cases? y/n ")
    
     if AllDataSets == 'y':
         BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
         BackGroundData.drop('EventID',axis=1,inplace=True)    
   
         for i in range(len(NEUTRALINOMASS)):
            Path = os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
            SignalEvents = pd.read_csv(Path)

        
            SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
            DataSet = pd.concat([BackGroundData,SignalEvents])
            DataSet.sample(frac=1)

            #DataSet = DataCuts(DataSet)           
         
            #RenameDataBaseColumns(DataSet)
         
            PCAPlots = PCAPlotter(DataSet,'Label')
            PCAPlots.PCAAnalysis()
         
     elif AllDataSets == 'n':
         InputsValid = False
         while InputsValid == False:
             try:
                 SmuonMass = int(input('Input the mass of the Smuon you want to investigate.'))
                 NeutralinoMass = int(input('Input the mass of the Neutralino you want to investigate.'))
                 InputsValid = True
             except:
                print("Inputs must be of type int.")
        
         BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
         BackGroundData.drop('EventID',axis=1,inplace=True)    
         try:
             Path = os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SmuonMass,NeutralinoMass))
             SignalEvents = pd.read_csv(Path)

         except: 
             print("Unable to locate Signal file.")
             raise FileNotFoundError
        
         SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
         DataSet = pd.concat([BackGroundData,SignalEvents])
         DataSet.sample(frac=1)

         #DataSet = DataCuts(DataSet)           
         
         #RenameDataBaseColumns(DataSet) #### THe renaming part needs to be redone so that we don't have to massive dictionaries in the file.
         ### Will need to fix this
         
         PCAPlots = PCAPlotter(DataSet,'Label', DataSet.columns)
         PCAPlots.PCAAnalysis()
            
             
             
     else:
         print('Input is not valid. Please enter either "y" or "n".')
         PCA_Plotter()
         
def Run_SHAP_PERM_TEST(DataSet, UseF1, paramList = None,Plot_titles=None):
       
    
    XGBModel = TreeModel(DataSet,ApplyDataCut=False, paramList=paramList) 
        
    XGBModel.XGBoostTrain(UseF1Score=UseF1)
    MeanSHAPValues = XGBModel.SHAPValuePlots(Plot_titles)
    
    
    PermValues = XGBModel.FeaturePermutation(usePredict_poba=False,Plot_Title=Plot_titles)
    
    MeanPermValues = {PermValues[i][-1] : PermValues[i][0] for i in range(len(PermValues))}
    
    return MeanSHAPValues, MeanPermValues
    
    
def SHAP_Perm_Test(Save_directory,UseF1):
    """
    SHAP Permutation test

    Returns
    -------
    None.

    """

    
    MeanSHAPValues = dict()
    MeanPermValues = dict()
    
    BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
    BackGroundData.drop('EventID',axis=1,inplace=True)    
   
    for i in range(len(NEUTRALINOMASS)):
        Path = os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
        SignalEvents = pd.read_csv(Path)
        
        SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
        DataSet = pd.concat([BackGroundData,SignalEvents])
        DataSet.sample(frac=1) 
        
        DataSet = DataCuts(DataSet)
        
        JSONParameters = RetrieveDictionary(HYPERPARAMETERLOCATION)
        try:
            paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])]
        except:
            paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
        
        MeanSHAPValues['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])], MeanPermValues['Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = Run_SHAP_PERM_TEST(DataSet, UseF1, paramList = paramList,Plot_titles='Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
    
    
    SaveDictionary(os.path.join(Save_directory,'Feature Dictionaries\MeanSHAPValues_No_ratios.txt'),MeanSHAPValues)
    
    SaveDictionary(os.path.join(Save_directory,'Feature Dictionaries\MeanPermValues_No_ratios.txt'),MeanPermValues)

def GeneralisabilityTest(ResultsLocation,UseF1):
    StopRun=False

    for i in range(len(SMUONMASS)):
        try:
            DataSet = pd.read_csv(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i])))
        except:
            print('Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{} is missing.'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
            StopRun=True
            
    if StopRun:
        print('Files missing stopping the run')
        os.stop()
    UseF1 = [UseF1] * len(SMUONMASS)   
    ResultsLocation = [ResultsLocation] * len(SMUONMASS)
    ZippedData = zip(SMUONMASS,NEUTRALINOMASS, UseF1, ResultsLocation)
    with Pool(NoofCPU) as pool:
         r = list(tqdm(pool.imap(runComparison,[Event for Event in ZippedData]),total=len(SMUONMASS)))
    
    pool.close()
    pool.join()
    
    CombinedDict = dict()
    
    print('Combining dictionary results')
    
    for i in tqdm(range(len(SMUONMASS))):
        FileName = 'Smuon_Mass_{}_Neutralino_{}_Scores.json'.format(SMUONMASS[i],NEUTRALINOMASS[i])
        Path = os.path.join(ResultsLocation[0],'Exclusion_plot_results')
        Temp = RetrieveDictionary(os.path.join(Path,FileName))  
        for keys in Temp.keys():
            CombinedDict[keys]=Temp[keys]
    SaveDictionary(os.path.join(ResultsLocation[0],'Model_Scores_AMS.json'),CombinedDict)
    if UseF1[0]:
        SaveDictionary(os.path.join(ResultsLocation[0],'Model_Scores_F1.json'),CombinedDict)
    else:
        SaveDictionary(os.path.join(ResultsLocation[0],'Model_Scores_AUC.json'),CombinedDict)

def AMSScoreTrainTest(ResultsLocation,UseF1):
    UseF1 = [UseF1] * len(SMUONMASS)   
    ResultsLocation = [ResultsLocation] * len(SMUONMASS)
    ZippedData = zip(SMUONMASS,NEUTRALINOMASS, UseF1, ResultsLocation)
    with Pool(NoofCPU) as pool:
         r = list(tqdm(pool.imap(runTrainingAMSScore,[Event for Event in ZippedData]),total=len(SMUONMASS)))
    
    pool.close()
    pool.join()
    
    CombinedDict = dict()
    
    print('Combining dictionary results')
    
    for i in tqdm(range(len(SMUONMASS))):
        FileName = 'Smuon_Mass_{}_Neutralino_{}_Scores.json'.format(SMUONMASS[i],NEUTRALINOMASS[i])
        Path = os.path.join(ResultsLocation[0],'AMS_Train_Test_Score')
        Temp = RetrieveDictionary(os.path.join(Path,FileName))  
        for keys in Temp.keys():
            CombinedDict[keys]=Temp[keys]
    if UseF1[0]:
        SaveDictionary(os.path.join(ResultsLocation[0],'AMS_Scores_F1.json'),CombinedDict)
    else:
        SaveDictionary(os.path.join(ResultsLocation[0],'AMS_Scores_AUC.json'),CombinedDict)

def Ratio_Test(UseF1):
    StopRun=False

    for i in range(len(SMUONMASS)):
        try:
            DataSet = pd.read_csv(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i])))
        except:
            print('Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{} is missing.'.format(SMUONMASS[i],NEUTRALINOMASS[i]))
            StopRun=True
            
    if StopRun:
        print('Files missing stopping the run')
        os.stop()
    
    Results = dict()
    ComparisonAUC = dict()
    ComparisonF1 = dict()
    for i in range(len(SMUONMASS)):
      BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
      BackGroundData.drop('EventID',axis=1,inplace=True)    

      SignalEvents = pd.read_csv(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i])))
      SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
      DataSet = pd.concat([BackGroundData,SignalEvents])
      DataSet.sample(frac=1)
      
      if UseF1:
          ComparisonF1['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = CompareModelwithandwithoutratios(DataSet,True)
      else:
          ComparisonAUC['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = CompareModelwithandwithoutratios(DataSet,False)
     
     
      DataSet = DataCuts(DataSet)
                
      RenameDataBaseColumns(DataSet)
            
      JSONParameters = RetrieveDictionary(HYPERPARAMETERLOCATION)
      try:
        paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)]
      except:
        paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
        
        
      #XGBModel = TreeModel(DataSet,paramList = paramList,ApplyDataCut=False) 
            
      #XGBModel.XGBoostTrain(UseF1Score=False)
      
           
      #Temp = XGBModel.FeaturePermutation(usePredict_poba=False)
      
      #XGBModel.SHAPValuePlots(Convert_to_Label('Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i]), Style = 1))
      
      #print(Results)
      #Results['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = dict(zip(Temp[:,3],Temp[:,0])) 

      #FeaturePermutationComparisonPlot(Results, PlotTitle='Permutation values for the features in the XGB model', YLabel ='Feature Permutation importance', YAxisTicks = None, Max_No_of_Labels = 10)
    if UseF1:
         FeaturePermutationComparisonPlot(ComparisonF1, PlotTitle='AMS score using the F1 metric', YLabel ='AMS score', YAxisTicks = None)
    else:
          FeaturePermutationComparisonPlot(ComparisonAUC, PlotTitle='AMS score using the AUC metric', YLabel ='AMS score', YAxisTicks = None)  

def Permutation_Plot_test(F1 =False):
    
    Results = dict()

    for i in range(len(SMUONMASS)):
      BackGroundData=pd.read_csv(os.path.join(CSVLOCATION,BACKGROUNDEVENTPATH))
      BackGroundData.drop('EventID',axis=1,inplace=True)    

      SignalEvents = pd.read_csv(os.path.join(CSVLOCATION,'Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i])))
      SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
      DataSet = pd.concat([BackGroundData,SignalEvents])
      DataSet.sample(frac=1)     
     
      DataSet = DataCuts(DataSet)
                
      RenameDataBaseColumns(DataSet)
            
      JSONParameters = RetrieveDictionary(HYPERPARAMETERLOCATION)
      try:
        paramList = JSONParameters['Smuon_Mass_{}_Neutralino_{}'.format(SMuon,Neutralino)]
      except:
        paramList = {'subsample': 1,
                     'reg_gamma': 0.4,
                     'reg_alpha': 0.1,
                     'n_estimators': 200,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
        
        
      XGBModel = TreeModel(DataSet,paramList = paramList,ApplyDataCut=False) 
            
      XGBModel.XGBoostTrain(UseF1Score=False)
      
           
      Temp = XGBModel.FeaturePermutation(usePredict_poba=False)
      
      XGBModel.SHAPValuePlots(Convert_to_Label('Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i]), Style = 1))
      
      print(Results)
      Results['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])] = dict(zip(Temp[:,3],Temp[:,0])) 

      FeaturePermutationComparisonPlot(Results, PlotTitle='Permutation values for the features in the XGB model', YLabel ='Feature Permutation importance', YAxisTicks = None, Max_No_of_Labels = 10)

########### Main execution 

if __name__ == "__main__":
    print("This script will execute all the processes needed to create all the plots in the paper.")
    ################TESTS TO RUN##############################
    Run_PCA_Plotter = False
    Run_SHAP_and_Permutation_tests = False
    Run_Generalisability_tests = False
    Run_ratio_and_HT_ST_comparison_tests = False
    Run_AUC_F1_AMS_comparison_test = False
    Run_Feature_Permutation_comparison_test = False
    AMSScoreTrainTest_check = False
    ################DIRECTORY FOR RESULTS#####################
    Directory_for_Results = r'G:\PPLM_results'
    ##########################################################
    print('The following tests will be run:')
    if Run_PCA_Plotter:
        print('PCA plotter.')
    if Run_SHAP_and_Permutation_tests:
        print("A test to determine SHAP and feature permutation values for each of the columns in the dataset.")
    if Run_Generalisability_tests:
        print("A test where a model will be created for one mass case and tested against the other mass cases  to determine how generalisable the model is to the other mass cases.")
    if Run_ratio_and_HT_ST_comparison_tests:
        print("A test to determine the effect of removing the feature engineered raios and the HT and ST features.")
    if Run_AUC_F1_AMS_comparison_test:
        print('Running test showing the AUC, F1 and AMS scores achieved for the diffirent mass cases.')
    if Run_Feature_Permutation_comparison_test:
        print("Running test comparing the feature permutation values for the diffirent mass cases.")
    if AMSScoreTrainTest_check:
        print('Running test for the AMS training score and the testing score.')
    print('Running tests for the following (Smuon, Nuetralino) mass cases:')
    print([i for i in zip(SMUONMASS,NEUTRALINOMASS)])
    if os.path.isdir(Directory_for_Results):
        print('Results will be saved in {}'.format(Directory_for_Results))
    else:
        print('The results directory is not accessible. Script will end excusion.')
        sys.exit()
    print('Testing if all files exist.')
    Test_for_Files()
    
    #### Begin functions ############
    if Run_PCA_Plotter:
        print('PCA plotter.')
        PCA_Plotter()
    if Run_SHAP_and_Permutation_tests:
        print("A test to determine SHAP and feature permutation values for each of the columns in the dataset.")
        SHAP_Perm_Test(Directory_for_Results,UseF1=False)
    if Run_Generalisability_tests:
        print("In this test we train the model using one of the mass cases and then apply the model to the other mass cases to determine the AMS, AUC and F1 scores for that mass case using the trained model.")
        GeneralisabilityTest(Directory_for_Results,True)
        
    if Run_ratio_and_HT_ST_comparison_tests:
        print('Run a test to determine the effect that the ratios and that the HT and ST variables hove on the efficiency of the models.')
        InputCheck = False
        while InputCheck == False:
            UseF1Score = input('Do you want to use only the AUC metric the F1 metric or both?F1/AUC/BOTH' )
            if UseF1Score == 'F1' :
                Ratio_Test(True)
                InputCheck = True
            elif UseF1Score == 'AUC':
                 Ratio_Test(False)
                 InputCheck = True
            elif UseF1Score == 'BOTH':
                 Ratio_Test(True)
                 Ratio_Test(False)
                 InputCheck = True
            else:
                print('Unrecognised input. Please type either "AUC", "F1" or "BOTH"')
                
        
    if Run_Feature_Permutation_comparison_test:
        print("Running test comparing the feature permutation values for the diffirent mass cases.")
        InputCheck = False
        while InputCheck == False:
            UseF1Score = input('Do you want to use only the AUC metric the F1 metric or both?F1/AUC/BOTH' )
            if UseF1Score == 'F1' :
                Permutation_Plot_test(True)
                InputCheck = True
            elif UseF1Score == 'AUC':
                 Permutation_Plot_test(False)
                 InputCheck = True
            elif UseF1Score == 'BOTH':
                 Permutation_Plot_test(True)
                 Permutation_Plot_test(False)
                 InputCheck = True
            else:
                print('Unrecognised input. Please type either "AUC", "F1" or "BOTH"')

    if AMSScoreTrainTest_check:
        AMSScoreTrainTest(Directory_for_Results,True)


def compareAllCases():
    BackGroundData=pd.read_csv(r'G:\CSV\Background_Events\EventData.csv')
    BackGroundData.drop('EventID',axis=1,inplace=True)    
   
    NEUTRALINOMASS=[270, 220, 190, 140, 130, 140, 95, 80, 60, 60, 65, 55, 200, 190, 180, 195, 96, 195, 96]
    SMUONMASS=[360, 320, 290, 240, 240, 420, 500, 400, 510, 200, 210, 250, 450, 500, 400, 400, 400, 200, 200]
    
    Results = dict()
    
    for i in range(len(NEUTRALINOMASS)):
        Path = 'G:\CSV\Events_PPtoSmuonSmuon_Smuon_Mass_{}_Neutralino_{}\EventData.csv'.format(SMUONMASS[i],NEUTRALINOMASS[i])
        SignalEvents = pd.read_csv(Path)
        SignalEvents.drop(['EventID'],axis=1,inplace=True)  
        
        DataSet = pd.concat([BackGroundData,SignalEvents])
        DataSet.sample(frac=1)
        
        Results['Smuon_Mass_{}_Neutralino_{}'.format(SMUONMASS[i],NEUTRALINOMASS[i])]=CompareModelwithandwithoutratios(DataSet)
        
    SaveDictionary('G:\Results For Particle Physics\Feature Dictionaries\AMSScoreResults\AMSScores_AUC_metric_With_weight_resampling.txt',Results)
    
    
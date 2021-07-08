# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:26:58 2020

@author: gerha
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from scipy.stats import uniform
import Feature_Plots_PCA
import shap
from PermImportance import PermulationImportance
import sys


def DictionaryPlot(DictList, ChartName):
    """
    Creates bar plots using a dictionary input.

    Parameters
    ----------
    DictList : Dict.
        Dictionary containing numerical values as its values, to be plotted.
    ChartName : Str.
        Title of the chart.

    Returns
    -------
    None.

    """
    plt.figure(figsize = (20, 20))
    plt.bar(range(len(DictList)),DictList.values()) 
    plt.title(ChartName)
    plt.ylabel('Relevance')
    plt.xticks(ticks = range(len(DictList)), labels = list(DictList.keys()), rotation=90)
    plt.show()
  
def ConfusionMatrixPlot(ConfusionResults, ListFeatures, ListCoeffs):
    fig = sns.heatmap(ConfusionResults, annot =True, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Model: XGBoost \n Features: {}".format(dict(zip(ListFeatures,ListCoeffs))))
    plt.plot(fig)
    plt.show()
    
def inverse_xgb_f1(y, t, threshold=0.5):
    t = t.get_label()
    y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
    Recall, Precision, F1, Avg = precision_recall_fscore_support(t, y_bin, beta=0.01, average = 'binary', zero_division = 0)
    #return 'f1',(1 - f1_score(t,y_bin))
    return 'f1',(1-F1)
    
def xgb_f1(y, t, threshold=0.5):
    t = t.get_label()
    y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
    Recall, Precision, F1, Avg = precision_recall_fscore_support(t, y_bin, beta=0.01, average = 'binary', zero_division = 0)
    #return 'f1',f1_score(t,y_bin)
    return 'f1', F1
    
def LabelClean(DataSet):
    try:
        DataSet.drop(['EventID'],axis=1,inplace=True)
    except:
        pass
    
    try: 
        DataSet.drop(['Label'],axis=1,inplace=True)
    except:
        pass
    return DataSet

def RemoveFeaturesNotinPaper(DataSet):
    RemoveColumns = ['PRI_leading_jet_pt',
      'PRI_subleading_jet_pt', 'PRI_leading_jet_eta',
      'PRI_subleading_jet_eta', 'PRI_lep_leading_pt', 'PRI_lep_subleading_pt',
      'PRI_lep_leading_eta', 'PRI_lep_subleading_eta', 'PRI_lep_leading_phi',
      'PRI_lep_subleading_phi', 'DER_P_T_ratio_lep_pair',
      'DER_Diff_Eta_lep_pair', 'DER_Diff_Phi_lep_pair', 'DER_sum_P_T',
      'PRI_Missing_pt']
    return DataSet.drop(RemoveColumns,axis=1)
    


def SumReturnDict(DataSet):
    Signal = sum(DataSet.Events_weight[DataSet.Label == 1])
    Background = sum(DataSet.Events_weight[DataSet.Label == 0])    
    return {'Signal' : Signal , 'Background' : Background, 'Ratio' : Signal/np.sqrt(Signal + Background)}

def DataCuts(DataSet, DisplayRemoval = False, SaveResults=False):
    """Function for introducing momentum and pseudorapidity cuts to the data """
    RemovalDict = dict()
    InitialWeights = dict()
    RemovalDict['Initial'] = SumReturnDict(DataSet)
    ######### Keep only dimuon events
    InitialWeight = sum(DataSet.Events_weight)
    DataSet = DataSet[DataSet.PRI_nMuons > 1]
    
    print("Initial weight {:.2f}".format(InitialWeight))
    print("New weight {:.2f}. Cut effciency {:.2f}% for lepton > 0 cut".format(sum(DataSet.Events_weight),(sum(DataSet.Events_weight)/InitialWeight)*100))
    RemovalDict['Cut l>0'] = SumReturnDict(DataSet)
    ############# Invariant muon mass cut ################
    InitialWeight = sum(DataSet.Events_weight)
    DataSet = DataSet[DataSet.DER_Muon_invariant_mass > 20]
    
    print("Initial weight {:.2f}".format(InitialWeight))
    print("New weight {:.2f}. Cut effciency {:.2f}% for Muon mass > 20 cut".format(sum(DataSet.Events_weight),(sum(DataSet.Events_weight)/InitialWeight)*100))
    RemovalDict['Cut Muon mass >20'] = SumReturnDict(DataSet)   
    
    ############## MT2 cuts ####################
    DataSet2 = DataSet.copy()
    InitialWeight = sum(DataSet.Events_weight)
    DataSet = DataSet[DataSet.DER_MT2_variable > 90]
    
    print("Initial weight {:.2f}".format(InitialWeight))
    print("New weight {:.2f}. Cut effciency {:.2f}% for MT_2 > 90 cut".format(sum(DataSet.Events_weight),(sum(DataSet.Events_weight)/InitialWeight)*100))
    RemovalDict['Cut l>0'] = SumReturnDict(DataSet)   
    
    if len(np.unique(DataSet.Label)) == 1:
        DataSet = DataSet2.copy()
        print("Unable to cut MT_2 variable.")
        import time 
        time.sleep(10)
    ####Clean the jet signals. To remove any soft jets.###
    InitialWeight = sum(DataSet.Events_weight)
    
    DataSet0 = DataSet[DataSet.PRI_jets == 0]
    DataSet1 = DataSet[(DataSet.PRI_jets == 1) & (DataSet.PRI_leading_jet_pt >= 25) & (abs(DataSet.PRI_leading_jet_eta) <= 2.5)]
    DataSet2 = DataSet[(DataSet.PRI_jets >= 2) & (DataSet.PRI_leading_jet_pt >= 25) & (abs(DataSet.PRI_leading_jet_eta) <= 2.5) & (DataSet.PRI_subleading_jet_pt >= 25) & (abs(DataSet.PRI_subleading_jet_eta) <= 2.5)]
    JetDataSet = pd.concat([DataSet0,DataSet1,DataSet2])
    
    print("Initial weight {:.2f}".format(InitialWeight))
    print("New weight {:.2f}. Cut effciency {:.2f}% for cut Jet_pt >= 25".format(sum(JetDataSet.Events_weight),(sum(JetDataSet.Events_weight)/InitialWeight)*100))

    
    RemovalDict['Cut PT(j) >= 25 and ETA(j) <= 2.5'] = SumReturnDict(JetDataSet)
    ### Clean the leptonic signals to remove any soft leptons####
    InitialWeight = sum(JetDataSet.Events_weight)
    
    DataSet3 = JetDataSet[JetDataSet.PRI_nleps == 0]
    DataSet4 = JetDataSet[(JetDataSet.PRI_nleps == 1) & (JetDataSet.PRI_lep_leading_pt >= 10) & (abs(JetDataSet.PRI_lep_leading_eta) <= 2.5)]
    DataSet5 = JetDataSet[(JetDataSet.PRI_nleps >= 2) & (JetDataSet.PRI_lep_leading_pt >= 10) & (abs(JetDataSet.PRI_lep_leading_eta) <= 2.5) & (JetDataSet.PRI_lep_subleading_pt >= 10) & (abs(JetDataSet.PRI_lep_subleading_eta) <= 2.5)]
    CleanedDataSet = pd.concat([DataSet3,DataSet4,DataSet5])

    print("Initial weight {:.2f}".format(InitialWeight))
    print("New weight {:.2f}. Cut effciency {:.2f}% for lepton cut >= 10".format(sum(CleanedDataSet.Events_weight),(sum(CleanedDataSet.Events_weight)/InitialWeight)*100))
    
    RemovalDict['Cut PT(l) >= 10 and ETA(l) <= 2.5'] = SumReturnDict(CleanedDataSet)
    
    #CleanedDataSet = CleanedDataSet[CleanedDataSet.HT > 120]
    
    #RemovalDict['THT > 120'] = SumReturnDict(CleanedDataSet)
    
    #CleanedDataSet = CleanedDataSet[CleanedDataSet.PRI_lep_leading_pt > 50]
    
    #RemovalDict['Leading Lepton > 50'] = SumReturnDict(CleanedDataSet)
    
    #InitialWeight = sum(CleanedDataSet.Events_weight)
    
    #CleanedDataSet = CleanedDataSet[CleanedDataSet.DER_PT_leading_lept_ratio_HT > 0.2]
    
    #print("Initial weight {:.2f}".format(InitialWeight))
    #print("New weight {:.2f}. Cut effciency {:.2f} for cut leading lepton momentum/H_T > 0.2%".format(sum(CleanedDataSet.Events_weight),(sum(CleanedDataSet.Events_weight)/InitialWeight)*100))
  
    
    #RemovalDict['Cut p_T^{l1}/H_T > 1.5'] =  SumReturnDict(CleanedDataSet)
    
    #InitialWeight = sum(CleanedDataSet.Events_weight)
     
    #CleanedDataSet = CleanedDataSet[CleanedDataSet.DER_PT_leading_lept_ratio_HT > 1]
 
    #print("Initial weight {:.2f}".format(InitialWeight))
    #print("New weight {:.2f}. Cut effciency {:.2f}% for cut pt^(l1)/H_T > 1.5".format(sum(CleanedDataSet.Events_weight),(sum(CleanedDataSet.Events_weight)/InitialWeight)*100))
 
    #RemovalDict['Cut p_T^{l1}/H_T > 1.5'] =  SumReturnDict(CleanedDataSet)
    

    ####CleanedDataSet = CleanedDataSet[CleanedDataSet.PRI_lep_leading_pt > 40]
    
    ####RemovalDict['PT_lep_leading > 50'] = SumReturnDict(CleanedDataSet)
    #######
    ####### Consider implementing this in a better way #######################
    #######
    if len(np.unique(CleanedDataSet.Label)) == 1:
        print('Dataset contains only one label.')
        print(np.unique(CleanedDataSet.Label))
        sys.exit('Dataset does not contain signal and background events.')
        
    return CleanedDataSet
    
class TreeModel():
    def __init__(self,DataSet,paramList = None, ApplyDataCut = True):
        if ApplyDataCut:
            DataSet = DataCuts(DataSet)
        #DataSet = RemoveFeaturesNotinPaper(DataSet)
        
        print("Orig : total weight sig", DataSet.Events_weight[DataSet.Label == 1].sum())
        print("Orig : total weight bkg", DataSet.Events_weight[DataSet.Label == 0].sum())
        try:
            self.df = DataSet.drop(['EventID','Label'],axis=1)
            self.Y = DataSet.Label
            self.labels = self.df.drop('Events_weight',axis=1).columns
        except:
            try:
                self.df = DataSet.drop(['Label'],axis=1)
                self.Y = DataSet.Label
                self.labels = self.df.drop('Events_weight',axis=1).columns  
            except:
                print('DataSet not of the correct form. Ensure that the particle'
                  / 'dataset has been produced by the ConvertLHEToTxt module')
                exit
        
        if paramList == None:
            self.HyperParameters = {}
        else:
            if 'base_score' in paramList:
                try: 
                    import click
                    if click.confirm('"base_score" does not work with the SHAP package, for SHAP version 0.37 and XGBoost version 1.3.0. Will do you wish to remove the feature? Doing so will allow SHAP to run otherwise it will not be run.'):
                        paramList.pop('base_score') 
                except:
                    print('base_score is not compatible with SHAP. Program will continue but may encounter errors when running SHAP.')

            self.HyperParameters = paramList
            
            
        self.TrainingTestSplit()
        

    def TrainingTestSplit(self):
        test_size = 0.3
        seed = 0
        X_train, X_test, y_train, y_test = train_test_split(self.df, self.Y, 
                                                            test_size=test_size, 
                                                            random_state=seed)
       
        self.TrainingData = pd.concat([X_train,y_train],axis =1)
        
        self.TestingData = pd.concat([X_test,y_test],axis=1)
        
        class_weights_train = (self.TrainingData.Events_weight[self.TrainingData.Label == 0].sum(), self.TrainingData.Events_weight[self.TrainingData.Label == 1].sum())

        for i in range(len(class_weights_train)):
            #training dataset: equalize number of background and signal
            self.TrainingData.loc[self.TrainingData.Label == i,'Events_weight'] *= max(class_weights_train)/ class_weights_train[i]
            #test dataset : increase test weight to compensate for sampling
            self.TestingData.loc[self.TestingData.Label == i,'Events_weight'] *= 1/(test_size)


        
        print ("Test : total weight sig", self.TestingData.Events_weight[self.TestingData.Label == 1].sum())
        print ("Test : total weight bkg", self.TestingData.Events_weight[self.TestingData.Label == 0].sum())
        print ("Train : total weight sig", self.TrainingData.Events_weight[self.TrainingData.Label == 1].sum())
        print ("Train : total weight bkg", self.TrainingData.Events_weight[self.TrainingData.Label == 0].sum())           
        
        
    

    def my_cv(self, df, predictors, response, kfolds, classifier, verbose=False):
        """Roll our own CV 
        train each kfold with early stopping
        return average metric, sd over kfolds, average best round"""
        EARLY_STOPPING_ROUNDS=100  # stop if no improvement after 100 rounds
        metrics = []
        best_iterations = []
        

        for train_fold, cv_fold in kfolds.split(df): 
            fold_X_train=df[predictors].values[train_fold]
            fold_y_train=df[response].values[train_fold]
            fold_X_test=df[predictors].values[cv_fold]
            fold_y_test=df[response].values[cv_fold]
            classifier.fit(fold_X_train, fold_y_train,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      eval_set=[(fold_X_test, fold_y_test)],
                      eval_metric=inverse_xgb_f1,
                      verbose=0,
                     )
            y_pred_test=classifier.predict(fold_X_test)
            y_pred_test = (y_pred_test > 0.5).astype(int)
            metrics.append(f1_score(fold_y_test,y_pred_test))
            best_iterations.append(classifier.best_iteration)
        return np.average(metrics), np.std(metrics), np.average(best_iterations)
        
    def cv_over_param_dict(self, df, param_dict, predictors, response, kfolds, verbose=False):
        """given a list of dictionaries of xgb params
        run my_cv on params, store result in array
        return updated param_dict, results dataframe
        """
        
        from datetime import datetime, timedelta
        
        BOOST_ROUNDS=50000
        start_time = datetime.now()
        print("%-20s %s" % ("Start Time", start_time))
        
        results = []
        
        Total = len(param_dict)
        
        for i, d in enumerate(param_dict):
            model = xgb.XGBClassifier(objective = "binary:logistic",
                                  n_estimators=BOOST_ROUNDS,
                                  verbosity=0,
                                  random_state=2012, 
                                  n_jobs=-1,
                                  booster='gbtree',
                                  eval_metric = inverse_xgb_f1,
                                  use_label_encoder=False,
                                  maximize = True,
                                  **d)
        
            metric, metric_std, best_iteration = self.my_cv(df, predictors, response, kfolds, model, verbose=False)    
            results.append([metric, metric_std, best_iteration, d])
            if i % 100 == 0:
                print("%s %3d of %3d result mean: %.6f std: %.6f, iter: %.2f" % (datetime.strftime(datetime.now(), "%T"), i, Total, metric, metric_std, best_iteration))
                if i % 1000 == 0 and i != 0:
                    DeltaTime = (datetime.now()-start_time).seconds
                    TimeperIter = round(DeltaTime/i)
                    EstimateLeft = timedelta(seconds=((Total - i) * TimeperIter))
                    print("%s %3d time taken: %s seconds per iteration: %s predicted time left: %s" % (datetime.strftime(datetime.now(), '%T'), i, str(timedelta(seconds=(datetime.now()-start_time).seconds)), TimeperIter, str(EstimateLeft)))
        end_time = datetime.now()
        print("%-20s %s" % ("Start Time", start_time))
        print("%-20s %s" % ("End Time", end_time))
        print(str(timedelta(seconds=(end_time-start_time).seconds)))
    
        results_df = pd.DataFrame(results, columns=['f1_score', 'std', 'best_iter', 'param_dict']).sort_values('f1_score',ascending=False)
        print(results_df.head())
    
        best_params = results_df.iloc[0]['param_dict']
        return best_params, results_df    


    def HyperParameterTuning(self, NoofTests = 200, No_jobs = -1):
        """
        Function for selecting the optimum values for the hyperparmeters to use for the XGBoost model.

        Parameters
        ----------
        DataSet : Pandas dataset
            Dataset containing the features to be used for training the model. There should be no weights included in this dataset.
        Y : One dimensional array
            Array or dataset containing the labels relating to the training dataset.

        Returns
        -------
        Dict.
            Returns a dictionary containing the optimum configuration of hyperparameters for creating the decision tree model.
            
            
        """
        from sklearn.model_selection import KFold 
        from itertools import product
        RANDOMSTATE = 2012
        
        NumberofTests = 8
        kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOMSTATE)
        
        current_params = {'subsample': 1,
                     'reg_alpha': 0.1,
                     'min_split_loss': 2,
                     'min_child_weight': 5,
                     'max_depth': 5,
                     'learning_rate': 0.1}
                
       	
        df = self.TrainingData.sample(frac=1)
        
        
        response = 'Label'
        predictors = df.drop(['Events_weight','Label'],axis=1).columns
        
        ##################################################
        # round 1: tune depth
        ##################################################
        max_depths = list(range(2,8))
        grid_search_dicts = [{'max_depth': md} for md in max_depths]
        # merge into full param dicts
        full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
        
        # cv and get best params
        current_params, results_df = self.cv_over_param_dict(df, full_search_dicts, predictors, response, kfolds)
        
                
        ##################################################
        # round 2: tune subsample, colsample_bytree, colsample_bylevel
        ##################################################
        subsamples = np.linspace(0.01, 1.0, NumberofTests)
        reg_alpha = np.linspace(0.1, 1.0, NumberofTests)
        min_child_weight = np.linspace(1,5,NumberofTests)
        min_split_loss = np.linspace(0,5,NumberofTests)
            
        grid_search_dicts = [dict(zip(['subsample', 'reg_alpha', 'min_child_weight', 'min_split_loss'], [a, b, c, d])) 
                             for a,b,c,d in product(subsamples, reg_alpha, min_child_weight, min_split_loss)]
        # merge into full param dicts
        full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
        # cv and get best params
        current_params, results_df = self.cv_over_param_dict(df, full_search_dicts, predictors, response, kfolds)
        
        # round 3: learning rate
        learning_rates = np.logspace(-3, -1, 5)
        grid_search_dicts = [{'learning_rate': lr} for lr in learning_rates]
        # merge into full param dicts
        full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]

        # cv and get best params
        current_params, results_df = self.cv_over_param_dict(df, full_search_dicts, predictors, response, kfolds, verbose=False)
        
        self.HyperParameters = current_params
        
              
        

    
    def ConfusionPairPlot(self,X_test,Y_test):
        """
        Creates a pair plot that colors the datapoints depending on if the points are True/False Positives/Negatives. This can be used to visualise the boundary cuts for the predictions.
        
        Parameters
        ----------
        X_test : Pandas DataFrame
            The dataframe sohuld contain the features required to perform predictions with the model.
        Y_test : One dimensional array
            Array containing the labels of the dataset.
        
        Returns
        -------
        Pairplot.
        
        """
        try:
            y_pred = self.Model.predict(X_test)
        except:
            print('Model does not work with a dataset, trying to use a DMatrix.')
            dtrain = xgb.DMatrix(X_test,Y_test)
            y_pred = self.Model.predict(dtrain)
            
            
        Predictions = [round(value) for value in y_pred]
        X_dev = X_test.copy()
        for i in tqdm(range(X_dev.shape[0])):
            if Predictions[i] == 0 and Y_test.iloc[i] == 0:
               Predictions[i] = 'True Negative'
            elif Predictions[i] == 0 and Y_test.iloc[i] == 1:
               Predictions[i] = 'False Negative'
            elif Predictions[i] == 1 and Y_test.iloc[i] == 0:
               Predictions[i] = 'False positive'
            elif Predictions[i] == 1 and Y_test.iloc[i] == 1:
               Predictions[i] = 'True positive'
    
        X_dev['Class'] = Predictions
        
        print('Predictions complete. Creating heat map plot.')
        try:
            Feature_Plots_PCA.FeaturePlots(X_dev, 'Class')
        except: 
            print('Some of the features do not have sufficient variation attempting to clean the dataset...')
            DropLabel = []
            for column in tqdm(X_dev.drop('Class',axis=1).columns):
                for Classes in X_dev['Class'].unique():
                    
                    if len(np.unique(X_dev[column][X_dev['Class'] == Classes ])) <= 2:
                        DropLabel.append(column)
                        print('For column {} of class {} contains the value {}.'.format(column,Classes,np.unique(X_dev[column][X_dev['Class'] == Classes ])))
                        print('This will cause an error with seaborn pairplot and so the column shall be removed.')
                        break
            
            try:
                Feature_Plots_PCA.FeaturePlots(X_dev.drop(np.unique(DropLabel),axis=1), 'Class')
            except:
                print('Unable to create pairplot to compare the results of the prediction.')
        print('Creating PCA plot.')
        #PCAPlots = Feature_Plots_PCA.PCAPlotter(X_dev,'Class')
        #PCAPlots.PCAAnalysis()
            
    def TreeDiagram(self):
        #fig, ax = plt.subplots(figsize=(300, 300))
        #plot_tree(self.Model, num_trees=0, ax=ax, rankdir = 'LR')
        #plt.show()
        for item in ['weight', 'gain', 'cover']:
            xgb.plot_importance(self.Model,
                                importance_type = item, 
                                title = 'Feature importance: {}'.format(item))
            plt.show()
    
    def XGBoostTrain(self, eval_metric = ['auc'], UseF1Score = False):
        """
        

        Parameters
        ----------
        eval_metric : string, optional
            DESCRIPTION. The default is 'auc'. For the f1 metric type 'f1'. You can add a list of evaluate metrics by listing them, note that the last one will be used as a training metric.

        Returns
        -------
        TYPE
            The trained model and is stored inthe 

        """
        # split data into train and test sets
        num_round = 2000
        EARLY_STOPPING = 10
        dtrain = xgb.DMatrix(self.TrainingData.drop(['Events_weight','Label'],axis =1),
                             label=self.TrainingData.Label,
                             weight=self.TrainingData.Events_weight)
        dtest = xgb.DMatrix(self.TestingData.drop(['Events_weight','Label'],axis =1),
                            label=self.TestingData.Label,
                            weight=self.TestingData.Events_weight )
        sum_wpos = sum( self.TrainingData.Events_weight.iloc[i] for i in range(len(self.TrainingData)) if self.TrainingData.Label.iloc[i] == 1.0  ) #This was used when the weights were not normalised.
        sum_wneg = sum( self.TrainingData.Events_weight.iloc[i] for i in range(len(self.TrainingData)) if self.TrainingData.Label.iloc[i] == 0.0  )
        #sum_wpos = len(self.TrainingData[self.TrainingData.Label==1]) 
        #sum_wneg = len(self.TrainingData[self.TrainingData.Label==0]) 
        print('Weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))
        paramList = self.HyperParameters
        if UseF1Score:
            paramList['disable_default_eval_metric'] = 1
            print('The f1 metric will be used to train the model.')
        else:
          paramList['eval_metric'] = eval_metric
                  
        paramList['tree_method'] ='hist'
        paramList['objective'] = 'binary:logistic'
        watchlist = [(dtrain,'train'), (dtest,'eval')]
        print('loading data end, start to boost trees')
        AdjustWeights = [0.0001,0.001,0.01,0.1,1]
        Results = []
        for i in AdjustWeights:
           paramList['scale_pos_weight'] = sum_wneg/sum_wpos * i
           if UseF1Score:
               self.Model = xgb.train(paramList,
                                      dtrain = dtrain,
                                      num_boost_round=100,
                                      feval = xgb_f1,
                                      evals = watchlist,
                                      early_stopping_rounds= EARLY_STOPPING,
                                      verbose_eval= False,
                                      maximize = True)
           else:
              self.Model = xgb.train(paramList,
                                     dtrain = dtrain,
                                     num_boost_round=100,
                                     evals = watchlist,
                                     early_stopping_rounds= EARLY_STOPPING,
                                     verbose_eval= False)
           Results.append(self.Model.best_score)
        print('Weight Adjust complete')
        print('Adjusting the weight by {}'.format(AdjustWeights[Results.index(max(Results))]))
        paramList['scale_pos_weight'] = sum_wneg/sum_wpos * AdjustWeights[Results.index(max(Results))]
        #paramList['scale_pos_weight'] = sum_wneg/sum_wpos * 0.00001
        if UseF1Score:
            self.Model = xgb.train(paramList,
                                   dtrain = dtrain,
                                   num_boost_round=num_round,
                                   feval = xgb_f1,
                                   evals = watchlist,
                                   early_stopping_rounds= EARLY_STOPPING,
                                   verbose_eval= True,
                                   maximize = True)
        else:
            self.Model = xgb.train(paramList,
                                   dtrain = dtrain,
                                   num_boost_round=num_round,
                                   evals = watchlist,
                                   early_stopping_rounds= EARLY_STOPPING,
                                   verbose_eval= True)
                
        
        self.ModelPredictions(self.TestingData,show=False)
        #self.Model.save_model('XGBoostModelFile')
        #self.TreeDiagram()
        #self.ConfusionPairPlot(self.TestingData.drop(['Events_weight','Label'],axis=1), self.TestingData.Label)
        return self.Model
    
    def ModelPredictions(self,DataSet,Metric='accuracy',show=True):
        DataSet1 = DataSet.copy()
        Y = DataSet1.Label
        DataSet1 = LabelClean(DataSet1)
        Matrix = xgb.DMatrix(DataSet1.drop('Events_weight',axis=1),label=Y,weight=DataSet.Events_weight)
        y_pred = self.Model.predict(Matrix)
        predictions = y_pred > 0.5
        if Metric == 'accuracy':
            accuracy = accuracy_score(Y, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
        elif Metric == 'f1':
            _, accuracy = xgb_f1(predictions, Matrix)
            print("F1 Score: %.2f%%" % (accuracy))            
        elif Metric == 'auc':
            accuracy = roc_auc_score(Matrix.get_label(), predictions)
        
        GXBoost_confusion = confusion_matrix(Y,predictions,normalize=None,sample_weight=DataSet1.Events_weight)
        print('{} events misclassified as true with an ams score of {}'.format(GXBoost_confusion[0,1], self.AMSScore(DataSet)))
        if show: 
            sns.heatmap(GXBoost_confusion,annot=True)
            plt.xlabel('Predicted value')
            plt.ylabel('True value')
            plt.title("XGBoost")
            plt.show()
        
        return accuracy
        
    def AMSScore(self,DataSet,Threshold=0.5):
        TotalSignal = sum(DataSet.Label)
        TotalBackground = len(DataSet) - TotalSignal
        DataSet1 = DataSet.copy()
        Y = DataSet1.Label
        DataSet1 = LabelClean(DataSet1)
        dtrain = xgb.DMatrix(DataSet.drop(['Events_weight','Label'],axis=1),Y)
        y_pred = self.Model.predict(dtrain)
        #Predictions = [round(value) for value in y_pred]
        Predictions = [(value > Threshold).astype(int) for value in y_pred]
        s = []
        b = []
        for i in range(y_pred.shape[0]):
            if Predictions[i] == 1 and Y.iloc[i] == 1:
               s.append(DataSet.Events_weight.iloc[i])
            elif Predictions[i] == 1 and Y.iloc[i] == 0:
               b.append(DataSet.Events_weight.iloc[i])
    
        print('Number of correctly classified signal events {}. Number of misclassified background events {}'.format(len(s),len(b)))
        try:
            print('Percentage of signal events correctly identified {:.2f}. Number of background events incorrectly identified {:.2f}'.format((len(s)/TotalSignal)*100,(len(b)/TotalBackground)*100))
        except: 
            if TotalSignal == 0:
                print('No signal found')
            elif TotalBackground == 0:
                print('No background found')
        signal = sum(s)
        background = sum(b)
        #signal = len(s)
        #background = len(b)
        if signal <= 0 :
                return 0
        elif background <= 0:
            return 10
        try:
            return signal/np.sqrt(background)
            #return np.sqrt(2*((signal+background)*np.log(1+float(signal)/background)-signal))
        except ValueError:
                print (1+float(signal)/background)
                print (2*((signal+background)*np.log(1+float(signal)/background)-signal))
            #return s/sqrt(s+b)
    
    def SHAPValuePlots(self,plt_Title=None):
        
        
        if 'base_score' not in self.HyperParameters:
            explainer = shap.TreeExplainer(self.Model)
            
            try: 
                DataSet = self.df.drop('Events_weight',axis=1)
            except:
                DataSet = self.df
        
            shap_values = explainer.shap_values(DataSet)
            MeanSHAP_values = dict(zip(DataSet.columns,np.abs(shap_values).mean(0)))
                           
            shap.force_plot(explainer.expected_value, shap_values[0,:], DataSet.columns,matplotlib=True)
            shap.summary_plot(shap_values, DataSet.columns, plot_type="bar",show=False)
            fig = plt.gcf()
            fig.suptitle(plt_Title, fontsize = 25)
            plt.show()
            xmax = shap_values.mean() + 3*shap_values.std()
            xmin = shap_values.mean() - 3*shap_values.std()
                
            List = list(np.unique(np.where(shap_values <= xmin)[0]))
            List = List + list(np.unique(np.where(shap_values >= xmax)[0]))
            
            DataSet = DataSet.drop(DataSet.index[List])
            shap_values = np.delete(shap_values,List,0)
            
            shap_values = explainer.shap_values(DataSet)
            shap.summary_plot(shap_values, DataSet,show=False)
            fig1 = plt.gcf()
            fig1.suptitle(plt_Title, fontsize = 25)
                      
            plt.show()
        
            return sorted(MeanSHAP_values, key=MeanSHAP_values.get,reverse=True)
            
    def FeaturePermutation(self, n_iterations=3,usePredict_poba=True, scoreFunction="AUC",Plot_Title=None):
        pi = PermulationImportance(model=self.Model,
                                   X=self.TestingData.drop(['Events_weight','Label'],axis =1),
                                   y=self.TestingData.Label,
                                   weights=self.TestingData.Events_weight,
                                   n_iterations=n_iterations,
                                   usePredict_poba=usePredict_poba,
                                   scoreFunction=scoreFunction)
        #pi.dislayResults()
        plt = pi.plotBars(PlotTitle = Plot_Title)
       
        plt.show()
        return pi.dislayResults()
      
        
# OLD AMS SCORE       
# np.sqrt(2*(s + b)*np.log(1 + s/b)-s)
def XGBoostersFeatureComparison(DataSet, Y):
    import itertools
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Y, test_size=test_size, random_state=seed)
    results = pd.DataFrame(columns=['num_features','features','Accuracy', 'ConfusionMatrix'])
    print("Training models")
    paramList = {'max_depth' : 6,
                     'nthread' : -1,
                     'tree_method' : 'gpu_hist',
                     'ojective' : 'binary:logistic',
                     'base_score' : 0.2,
                     'alpha' : 1 }
    model = xgb.XGBClassifier(**paramList)
    if len(np.unique(Y)) == 2:
      for k in range(1, X_train.shape[1] + 1):
        for subset in tqdm(itertools.combinations(range(X_train.shape[1]), k), leave = None):
            subset = list(subset)
            model.fit(X = X_train[X_train.columns[subset]], y = y_train, eval_metric = "ams@0" ,eval_set = [(X_test[X_test.columns[subset]], y_test)], verbose = False, early_stopping_rounds = 10)
            y_pred = model.predict(X_test[X_test.columns[subset]])
            predictions = [value for value in y_pred]
            accuracy = accuracy_score(y_test, predictions)
            XGBoost_confusion = confusion_matrix(y_test, y_pred, normalize = 'true')
            results = results.append(pd.DataFrame([{'num_features' : k,
                                                  'features' : DataSet.columns[subset],
                                                  'coeffs' : np.round(model.feature_importances_),
                                                  'Accuracy' : accuracy,
                                                  'ConfusionMatrix' : XGBoost_confusion}]))   
            
    else: 
        for k in range(1, X_train.shape[1] + 1):
            for subset in tqdm(itertools.combinations(range(X_train.shape[1]), k), leave = None):
                subset = list(subset)  
                model.fit(X = X_train, y = y_train, eval_metric = "ams@0" ,eval_set = [(X_test, y_test)], verbose = True, early_stopping_rounds = 10)   
                y_pred = model.predict(X_test[X_test.columns[subset]])
                predictions = [round(value) for value in y_pred]
                accuracy = accuracy_score(y_test, predictions)
                XGBoost_confusion = confusion_matrix(y_test, y_pred, normalize = 'true')
                results = results.append(pd.DataFrame([{'num_features' : k,
                                                  'features' : DataSet.columns[subset],
                                                  'coeffs' : np.round(np.concatenate((model.feature_importances_))),
                                                  'Accuracy' : accuracy,
                                                  'ConfusionMatrix' : XGBoost_confusion}]))
    results = results.sort_values('Accuracy').reset_index()
    BestScore = results['Accuracy'].max()
    Features = {i : 0 for i in DataSet.columns}
    for x in range(len(results)):
        if results['Accuracy'][x] == BestScore:
            if all(results['coeffs'][x]) != 0:
                ConfusionMatrixPlot(results['ConfusionMatrix'][x], results['features'][x], results['coeffs'][x])
                
            #DictionaryPlot(dict(zip(results['features'][x].tolist(),results['coeffs'][x].tolist())), 'Logistic Linear Regression with accuracy {}'.format(results['Accuracy'][x]))
            #print('Features for top result :{}'.format(dict(zip(results['features'][x].tolist(),results['coeffs'][x].tolist()))))
            for i in range(len(results['features'][x])):
                if results['coeffs'][x][i] != 0:
                   Features[results['features'][x][i]] = Features[results['features'][x][i]] + 1
                
    Features = {k: v for k, v in sorted(Features.items(), key = lambda item: item[1], reverse = True)}  
    print(Features)  
    DictionaryPlot(Features, "Frequency of features in best models for XGBoost")
    return results 

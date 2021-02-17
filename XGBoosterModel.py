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
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import uniform
import Feature_Plots_PCA
import shap
from PermImportance import PermulationImportance


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
    
def xgb_f1(y, t, threshold=0.1):
    t = t.get_label()
    y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
    return 'f1', f1_score(t,y_bin)

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
    


def DataCuts(DataSet, DisplayRemoval = False):
    """Function for introducing momentum and pseudorapidity cuts to the data """
    ####Clean the jet signals. To remove any soft jets.###
    DataSet0 = DataSet[DataSet.PRI_jets == 0]
    DataSet1 = DataSet[(DataSet.PRI_jets == 1) & (DataSet.PRI_leading_jet_pt >= 25) & (abs(DataSet.PRI_leading_jet_eta) <= 2.5)]
    DataSet2 = DataSet[(DataSet.PRI_jets >= 2) & (DataSet.PRI_leading_jet_pt >= 25) & (abs(DataSet.PRI_leading_jet_eta) <= 2.5) & (DataSet.PRI_subleading_jet_pt >= 25) & (abs(DataSet.PRI_subleading_jet_eta) <= 2.5)]
    if DisplayRemoval:
        print('{} events removed from the dataset as the jets had a momentum lower than 25GeV or the psuedorapidity values of the jets was greater than 2.5.'.format(len(DataSet)-len(pd.concat([DataSet0,DataSet1,DataSet2]))))
    JetDataSet = pd.concat([DataSet0,DataSet1,DataSet2])
    
    ### Clean the leptonic signals to remove any soft leptons####
    DataSet3 = JetDataSet[JetDataSet.PRI_nleps == 0]
    DataSet4 = JetDataSet[(JetDataSet.PRI_nleps == 1) & (JetDataSet.PRI_lep_leading_pt >= 10) & (abs(JetDataSet.PRI_lep_leading_eta) <= 2.5)]
    DataSet5 = JetDataSet[(JetDataSet.PRI_nleps >= 2) & (JetDataSet.PRI_lep_leading_pt >= 10) & (abs(JetDataSet.PRI_lep_leading_eta) <= 2.5) & (JetDataSet.PRI_lep_subleading_pt >= 10) & (abs(JetDataSet.PRI_lep_subleading_eta) <= 2.5)]
    if DisplayRemoval:
        print('{} events removed from the dataset as the leptons had a momentum less than 10GeV, or had a pseudorapidity of greater than 2.5. '.format(len(JetDataSet)-len(pd.concat([DataSet3,DataSet4,DataSet5]))))
    CleanedDataSet = pd.concat([DataSet3,DataSet4,DataSet5])
    
    #######
    ####### Consider implementing this in a better way #######################
    #######
    # CleanedDataSet = RemoveFeaturesNotinPaper(CleanedDataSet)
    ######
    ###### This function might have to be removed ############################
    ######
    return CleanedDataSet
    
class TreeModel():
    def __init__(self,DataSet,paramList = None, ApplyDataCut = True):
        if ApplyDataCut:
            DataSet = DataCuts(DataSet)
        #DataSet = RemoveFeaturesNotinPaper(DataSet)
        
        print ("Orig : total weight sig", DataSet.Events_weight[DataSet.Label == 1].sum())
        print ("Orig : total weight bkg", DataSet.Events_weight[DataSet.Label == 0].sum())
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
        
        
        
        


    def HyperParameterTuning(self, NoofTests = 200):
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
                
        try:
        	DataSet = self.TrainingData.sample(n=50000)
        except:
        	DataSet = self.TrainingData	
        
        Labels = DataSet.Label
        
        #SHAP does not for some reason work with the base score feature.
        param_grid =  { 'learning_rate' : [0.01, 0.1, 0.5, 0.9],
                       'n_estimators' : [10, 50, 100, 150, 200],
                       'subsample': [0.3, 0.5, 0.9],
                       'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       #'base_score' : [0.1, 0.5, 0.9],
                       'reg_alpha' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                       'min_split_loss' : [0, 0.5, 0.8, 1, 2],
                       'reg_gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ],
                       'min_child_weight' : range(5,8)}
        #### We are goiing to keep some parameters constant during this test. If you wish to test all parameters delete the below paramgrid.
        param_grid =  { 'learning_rate' : uniform(loc=0.01, scale=1), ###Uniform creates a distribution of [loc, loc + scale]
                       'n_estimators' : [200],
                       'subsample': uniform(loc=0.8,scale=0.2),
                       'max_depth': list(range(4,11)),
                       'reg_alpha' : uniform(0,0.6),
                       'min_split_loss' : [0, 0.5, 0.8, 1, 2],
                       'reg_gamma' : [0, 1, 5],
                       'min_child_weight' : [5]}
        model = xgb.XGBClassifier(objective = "binary:logistic", verbosity=0,use_label_encoder=False)
        randomized_mse = RandomizedSearchCV(estimator = model, 
                                            param_distributions=param_grid,
                                            n_iter = NoofTests, scoring='f1', 
                                            n_jobs =-1, cv=4, verbose = 1)
        randomized_mse.fit(DataSet.drop(['Events_weight','Label'],axis=1), Labels)
        print('Best parameters found: ', randomized_mse.best_params_)
        print('Best accuracy found: ', np.sqrt(np.abs(randomized_mse.best_score_)))
        self.HyperParameterResults = randomized_mse
        self.HyperParameters = randomized_mse.best_params_
        return randomized_mse.best_params_
        
              
        

    
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
        for i in range(X_dev.shape[0]):
            if Predictions[i] == 0 and Y_test.iloc[i] == 0:
               Predictions[i] = 'True Negative'
            elif Predictions[i] == 0 and Y_test.iloc[i] == 1:
               Predictions[i] = 'False Negative'
            elif Predictions[i] == 1 and Y_test.iloc[i] == 0:
               Predictions[i] = 'False positive'
            elif Predictions[i] == 1 and Y_test.iloc[i] == 1:
               Predictions[i] = 'True positive'
    
        X_dev['Class'] = Predictions
        
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
        
        PCAPlots = Feature_Plots_PCA.PCAPlotter(X_dev,'Class')
        PCAPlots.PCAAnalysis()
            
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
        num_round = 200
        dtrain = xgb.DMatrix(self.TrainingData.drop(['Events_weight','Label'],axis =1),
                             label=self.TrainingData.Label,
                             weight=self.TrainingData.Events_weight)
        dtest = xgb.DMatrix(self.TestingData.drop(['Events_weight','Label'],axis =1),
                            label=self.TestingData.Label,
                            weight=self.TestingData.Events_weight )
        sum_wpos = sum( self.TrainingData.Events_weight.iloc[i] for i in range(len(self.TrainingData)) if self.TrainingData.Label.iloc[i] == 1.0  )
        sum_wneg = sum( self.TrainingData.Events_weight.iloc[i] for i in range(len(self.TrainingData)) if self.TrainingData.Label.iloc[i] == 0.0  )
        print ('Weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))
        paramList = self.HyperParameters
        if UseF1Score:
            paramList['disable_default_eval_metric'] = 1
            print('The f1 metric will be used to train the model.')
        else:
          paramList['eval_metric'] = eval_metric
                  
        paramList['tree_method'] ='hist'
        paramList['objective'] = 'binary:logistic'
        watchlist = [(dtrain,'train'), (dtest,'eval')]
        print ('loading data end, start to boost trees')
        AdjustWeights = [0,0.001,0.01,0.1,1]
        Results = []
        for i in AdjustWeights:
            paramList['scale_pos_weight'] = sum_wneg/sum_wpos * i
            if UseF1Score:
                self.Model = xgb.train(paramList, dtrain = dtrain,num_boost_round=num_round, feval = xgb_f1, evals = watchlist, early_stopping_rounds= 50, verbose_eval= False,  maximize = True)
            else:
                self.Model = xgb.train(paramList, dtrain = dtrain,num_boost_round=num_round, evals = watchlist, early_stopping_rounds= 50, verbose_eval= False)
            Results.append(self.Model.best_score)
        print('Weight Adjust complete')
        print('Adjusting the weight by {}'.format(AdjustWeights[Results.index(max(Results))]))
        paramList['scale_pos_weight'] = sum_wneg/sum_wpos * AdjustWeights[Results.index(max(Results))]
        
        if UseF1Score:
            self.Model = xgb.train(paramList, dtrain = dtrain,num_boost_round=num_round, feval = xgb_f1, evals = watchlist, early_stopping_rounds= 50, verbose_eval= True, maximize = True)
        else:
            self.Model = xgb.train(paramList, dtrain = dtrain,num_boost_round=num_round, evals = watchlist, early_stopping_rounds= 50, verbose_eval= True)
             
        
        self.ModelPredictions(self.TestingData)
        self.Model.save_model('XGBoostModelFile')
        self.TreeDiagram()
        #self.ConfusionPairPlot(self.TestingData.drop(['Events_weight','Label'],axis=1), self.TestingData.Label)
        return self.Model
    
    def ModelPredictions(self,DataSet,Metric='accuracy'):
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
        
        GXBoost_confusion = confusion_matrix(Y,predictions,normalize=None)
        print('{} events misclassified as true with an ams score of {}'.format(GXBoost_confusion[0,1], self.AMSScore(DataSet)))
        
        sns.heatmap(GXBoost_confusion,annot=True)
        plt.xlabel('Predicted value')
        plt.ylabel('True value')
        plt.title("XGBoost")
        plt.show()
        
        return accuracy
        
    def AMSScore(self,DataSet):
        DataSet1 = DataSet.copy()
        Y = DataSet1.Label
        DataSet1 = LabelClean(DataSet1)
        dtrain = xgb.DMatrix(DataSet.drop(['Events_weight','Label'],axis=1),Y)
        y_pred = self.Model.predict(dtrain)
        Predictions = [round(value) for value in y_pred]
        s = []
        b = []
        for i in range(y_pred.shape[0]):
            if Predictions[i] == 1 and Y.iloc[i] == 1:
               s.append(DataSet.Events_weight.iloc[i])
            elif Predictions[i] == 1 and Y.iloc[i] == 0:
               b.append(DataSet.Events_weight.iloc[i])
    
        print('Number of correctly classified signal events {}. Number of misclassified background events {}'.format(len(s),len(b)))
        s = sum(s)
        b = sum(b)
        if b<=0 or s<=0:
                return 0
        try:
            return np.sqrt(2*((s+b)*np.log(1+float(s)/b)-s))
        except ValueError:
                print (1+float(s)/b)
                print (2*((s+b)*np.log(1+float(s)/b)-s))
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

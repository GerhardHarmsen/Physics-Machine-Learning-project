# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:29 2020

@author: gerha
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import textwrap 
import click

TitleSize = 30
FontSize  = 25
plt.rc('legend',fontsize=30)

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                                and (abs(i[1]-x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height:
                differ = np.diff(sorted_ltp, axis = 0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    if j > txt_height * 1.5:
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text_positions, labels, axis,txt_width,txt_height):
    i = 0
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - .03, 1.02*t, labels[i],rotation=0, color='black', fontsize=FontSize)
        i = i + 1
        if y != t:
            axis.arrow(x, t,0,y-t, color='black',alpha=0.2, width=0.014,
                       head_width=.02, head_length=txt_height*0.5,
                       zorder=0,length_includes_head=True)

def PLTColumns(column1, column2):
  XData = []
  YData = []
  for i in range(dtrain.shape[0]):
      if dtrain[i,column1] != -999 and dtrain[i,column2] != -999:
         XData.append(dtrain[i,column1]) 
         YData.append(dtrain[i,column2])
  fig, ax = plt.subplots()
  fig = plt.scatter(XData,YData);
  ax.set_xlabel(xlabel = columnNames[column1], fontsize = FontSize)
  ax.set_ylabel(ylabel = columnNames[column2], fontsize = FontSize)
  plt.show()

def FeaturePlots(DataSet, LabelOfInterest):
    sns.pairplot(DataSet, hue = LabelOfInterest)
    plt.title('Linear correlation plot of the features in the dataset')
    plt.show()
    sns.heatmap(DataSet.corr(), xticklabels=DataSet.columns, yticklabels=DataSet.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = False)
    plt.title('Heat map of the features showing linear correlation of the features')
    plt.show()

class PCAPlotter():
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
        
        
        
        
        
    def DrawBoxPlots(self):
        if self.ContainsWeights:
            DataSet2 = self.df.drop(labels = [self.lbl, 'Events_weight'], axis = 1)
        else:
            DataSet2 = self.df.drop(labels = self.lbl, axis = 1)
            
        for col in np.unique(DataSet2.columns):
            plt.figure(num =None, figsize = [20, 20])
            boxplot1 = self.df.boxplot(by = self.lbl, column = col)
            plt.show()        

    def PCAAnalysis(self, MinNoofJets = 0, MinNoofLeptons = 0, MaxNoofJets = 2, MaxNoofLeptons = 2, ShowPlots = True):
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
       
        NoofColumns = (MaxNoofLeptons - MinNoofLeptons) + 1
        NoofRows = (MaxNoofJets - MinNoofJets) + 1
        
           
        PCADataSet = self.df[self.df.PRI_jets >= MinNoofJets]
        PCADataSet = PCADataSet[PCADataSet.PRI_nleps >= MinNoofLeptons]
        
        self.FeaturePCAValues = {"Leptons {} Jets {}".format(i,j) : {} for i,j in itertools.product(range(MinNoofLeptons,MaxNoofLeptons + 1),range(MinNoofJets,MaxNoofJets + 1))}
        self.FeaturePCAPercentage = {"Leptons {} Jets {}".format(i,j) : {} for i,j in itertools.product(range(MinNoofLeptons,MaxNoofLeptons + 1),range(MinNoofJets,MaxNoofJets + 1))}
        
        if (MaxNoofJets == 0) and (MaxNoofLeptons == 0):
           exit('Unable to create reasonable PCA plot. The max number of jets or leptons must be atleast one.')
           
        if NoofColumns == 1 and NoofRows == 1:
            self.OnePlot(MinNoofJets, MinNoofLeptons, PCADataSet)
            
        elif NoofColumns == 1 and NoofRows > 1:
            self.JetsPlots(MinNoofJets, MaxNoofJets, MinNoofLeptons, PCADataSet)
            
        elif NoofColumns > 1 and NoofRows == 1:
            self.LeptonsPlots(MinNoofJets, MinNoofLeptons, MaxNoofLeptons, PCADataSet)
            
        elif NoofColumns > 1 and NoofRows > 1:
            self.JetsandLeptonsPlot(MinNoofJets, MinNoofLeptons, MaxNoofJets, MaxNoofLeptons, PCADataSet)
           
           
        plt.tight_layout() #This to avoid overlap of labels and titles across plots
        if ShowPlots: plt.show()
        

    def OnePlot(self, NoofJets, NoofLeptons, PCADataSet):
        fig, axes = plt.subplots(nrows = 1, ncols = 1 , figsize=(40, 40))
        fig1, axes1 = plt.subplots(nrows = 1, ncols = 1, figsize=(40, 40))
        fig2, axes2 = plt.subplots(nrows = 1, ncols = 1, figsize=(40, 40))
        
        self.PCAPlots(PCADataSet,NoofJets,NoofLeptons, ax=axes,ax1=axes1, ax2=axes2)      
                       
        axes.legend()
        PercentageinPlot = len(PCADataSet) / len(self.df) * 100
        axes.set_title('Percentage of the total dataset: {} %'.format(round(PercentageinPlot)),fontsize = TitleSize)
        axes.set_xlabel('Between {} and {} leptons'.format(NoofLeptons,max(PCADataSet.PRI_nleps)),fontsize = FontSize)
        axes.set_ylabel('Between {} and {} Jets'.format(NoofJets,max(PCADataSet.PRI_jets) ), fontsize = FontSize)
        axes1.set_xlabel('Between {} and {} leptons'.format(NoofLeptons,max(PCADataSet.PRI_nleps)),fontsize = FontSize)
        axes1.set_ylabel('Between {} and {} Jets'.format(NoofJets,max(PCADataSet.PRI_jets) ), fontsize = FontSize)
        
    def LeptonsPlots(self, NoofJets, MinNoofLeptons, MaxNoofLeptons, PCADataSet):
        
        NoofColumns = (MaxNoofLeptons - MinNoofLeptons) + 1
        
        fig, axes = plt.subplots(nrows = 1, ncols = NoofColumns , figsize=(40 * NoofColumns, 40))
        fig1, axes1 = plt.subplots(nrows = 1, ncols = NoofColumns, figsize=(40 * NoofColumns, 40))
        fig2, axes2 = plt.subplots(nrows = 1, ncols = NoofColumns, figsize=(40 * NoofColumns, 40))
        
        for Leptons in range(NoofColumns):
             if (Leptons + MinNoofLeptons) == (MaxNoofLeptons):
                  self.PCAPlots(PCADataSet[PCADataSet.PRI_nleps >= (Leptons + MinNoofLeptons)],NoofJets,(Leptons + MinNoofLeptons), ax=axes[Leptons],ax1=axes1[Leptons],ax2=axes2[Leptons])                          
                
                  PercentageinPlot = len(PCADataSet[PCADataSet.PRI_nleps >= (Leptons + MinNoofLeptons)]) / len(self.df) * 100
             else:
                 self.PCAPlots(PCADataSet[PCADataSet.PRI_nleps == (Leptons + MinNoofLeptons)],NoofJets,(Leptons + MinNoofLeptons), ax=axes[Leptons],ax1=axes1[Leptons],ax2=axes2[Leptons])
             
                 PercentageinPlot = len(PCADataSet[PCADataSet.PRI_nleps >= (Leptons + MinNoofLeptons)]) / len(self.df) * 100   
                                          
             
               
             axes[Leptons].legend()
             axes[Leptons].set_title('Percentage of the total dataset: {} %'.format(round(PercentageinPlot)),fontsize = TitleSize)
             axes[Leptons].set_xlabel('Number of leptons: {}'.format(Leptons + MinNoofLeptons),fontsize = FontSize)
             axes[0].set_ylabel('Number of Jets: {}'.format(NoofJets), fontsize = FontSize)
             axes1[Leptons].set_xlabel('Number of leptons: {}'.format(Leptons + MinNoofLeptons),fontsize = FontSize)
             axes1[0].set_ylabel('Number of Jets: {}'.format(NoofJets), fontsize = FontSize)
             
        axes[NoofColumns - 1].set_xlabel('Between {} and {} leptons'.format(MaxNoofLeptons,max(PCADataSet.PRI_nleps)),fontsize = FontSize)
        axes1[NoofColumns - 1].set_xlabel('Between {} and {} leptons'.format(MaxNoofLeptons,max(PCADataSet.PRI_nleps)),fontsize = FontSize)
      
    def JetsPlots(self, MinNoofJets, MaxNoofJets, NoofLeptons, PCADataSet):
         NoofColumns = (MaxNoofJets - MinNoofJets) + 1
        
         fig, axes = plt.subplots(nrows = 1, ncols = NoofColumns , figsize=(40 * NoofColumns, 40))
         fig1, axes1 = plt.subplots(nrows = 1, ncols = NoofColumns, figsize=(40 * NoofColumns, 40))
         fig2, axes2 = plt.subplots(nrows = 1, ncols = NoofColumns, figsize=(40 * NoofColumns, 40))
         
         for Jets in range(NoofColumns):
             if (Jets + MinNoofJets) == (MaxNoofJets):
                   self.PCAPlots(PCADataSet[PCADataSet.PRI_jets >= (Jets + MinNoofJets)],(Jets + MinNoofJets), NoofLeptons, ax=axes[Jets],ax1=axes1[Jets],ax2=axes2[Jets])
                 
                   PercentageinPlot = len(PCADataSet[PCADataSet.PRI_jets >= (Jets + MinNoofJets)]) / len(self.df) * 100
             else:
                    self.PCAPlots(PCADataSet[PCADataSet.PRI_jets == (Jets + MinNoofJets)],(Jets + MinNoofJets), NoofLeptons, ax=axes[Jets],ax1=axes1[Jets],ax2=axes2[Jets])
                  
                    PercentageinPlot = len(PCADataSet[PCADataSet.PRI_jets == (Jets + MinNoofJets)]) / len(self.df) * 100 
            
              
             axes[Jets].legend()
             axes[Jets].set_title('Percentage of the total dataset: {} %'.format(round(PercentageinPlot)),fontsize = TitleSize)
             axes[Jets].set_xlabel('Number of jets: {}'.format((Jets + MinNoofJets)),fontsize = FontSize)
             axes[0].set_ylabel('Number of leptons: {}'.format(NoofLeptons), fontsize = FontSize)
             axes1[Jets].set_xlabel('Number of jets: {}'.format((Jets + MinNoofJets)),fontsize = FontSize)
             axes1[0].set_ylabel('Number of leptons: {}'.format(NoofLeptons), fontsize = FontSize)
             
         axes[NoofColumns - 1].set_xlabel('Between {} and {} jets'.format(MaxNoofJets,max(PCADataSet.PRI_jets)),fontsize = FontSize)
         axes1[NoofColumns - 1].set_xlabel('Between {} and {} jets'.format(MaxNoofJets,max(PCADataSet.PRI_jets)),fontsize = FontSize)

    def JetsandLeptonsPlot(self, MinNoofJets, MinNoofLeptons, MaxNoofJets, MaxNoofLeptons, PCADataSet):
        NoofColumns = (MaxNoofLeptons - MinNoofLeptons) + 1
        NoofRows = (MaxNoofJets - MinNoofJets) + 1
        
        fig, axes = plt.subplots(nrows = NoofRows, ncols = NoofColumns , figsize=(40, 40))
        fig1, axes1 = plt.subplots(nrows = NoofRows, ncols = NoofColumns, figsize=(40, 40))
        fig2, axes2 = plt.subplots(nrows = NoofRows, ncols = NoofColumns, figsize=(40, 40))
        
        for Jets in range(NoofRows):
            for Leptons in range(NoofColumns):
                if (Jets + MinNoofJets) == (MaxNoofJets):
                       if (Leptons + MinNoofLeptons) == (MaxNoofLeptons):
                         self.PCAPlots(PCADataSet[(PCADataSet.PRI_nleps >= (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets >= (Jets + MinNoofJets))],
                                      (Jets + MinNoofJets),
                                      (Leptons + MinNoofLeptons),
                                      ax=axes[Jets,Leptons],
                                      ax1=axes1[Jets,Leptons],
                                      ax2=axes2[Jets,Leptons])
                         
                         PercentageinPlot = len(PCADataSet[(PCADataSet.PRI_nleps >= (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets >= (Jets + MinNoofJets))]) / len(self.df) * 100
                           
                          
                       else:
                         self.PCAPlots(PCADataSet[(PCADataSet.PRI_nleps == (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets >= (Jets + MinNoofJets))],
                                      (Jets + MinNoofJets),
                                      (Leptons + MinNoofLeptons),
                                      ax=axes[Jets,Leptons],
                                      ax1=axes1[Jets,Leptons],
                                      ax2=axes2[Jets,Leptons]) 
                         
                         PercentageinPlot = len(PCADataSet[(PCADataSet.PRI_nleps == (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets >= (Jets + MinNoofJets))]) / len(self.df) * 100
                                     
                else:
                     if (Leptons + MinNoofLeptons) == (MaxNoofLeptons):
                         self.PCAPlots(PCADataSet[(PCADataSet.PRI_nleps >= (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets == (Jets + MinNoofJets))],
                                      (Jets + MinNoofJets),
                                      (Leptons + MinNoofLeptons),
                                      ax=axes[Jets,Leptons],
                                      ax1=axes1[Jets,Leptons],
                                      ax2=axes2[Jets,Leptons]) 
                         
                         PercentageinPlot = len(PCADataSet[(PCADataSet.PRI_nleps >= (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets == (Jets + MinNoofJets))]) / len(self.df) * 100
                           
                          
                     else:
                         self.PCAPlots(PCADataSet[(PCADataSet.PRI_nleps == (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets == (Jets + MinNoofJets))],
                                      (Jets + MinNoofJets),
                                      (Leptons + MinNoofLeptons),
                                      ax=axes[Jets,Leptons],
                                      ax1=axes1[Jets,Leptons],
                                      ax2=axes2[Jets,Leptons])   
                         
                         PercentageinPlot = len(PCADataSet[(PCADataSet.PRI_nleps == (Leptons + MinNoofLeptons)) & (PCADataSet.PRI_jets == (Jets + MinNoofJets))]) / len(self.df) * 100
            
                      
                                 
                axes[Jets,Leptons].legend()
                axes[Jets,Leptons].set_title('Percentage of the total dataset: {} %'.format(round(PercentageinPlot)),fontsize = TitleSize)
                axes[NoofRows - 1,Leptons].set_xlabel('Number of leptons: {}'.format(Leptons+MinNoofLeptons),fontsize = FontSize)
                axes[Jets,0].set_ylabel('Number of Jets: {}'.format(Jets + MinNoofJets), fontsize = FontSize)
                axes1[NoofRows - 1,Leptons].set_xlabel('Number of leptons: {}'.format(Leptons + MinNoofLeptons),fontsize = FontSize)
                axes1[Jets,0].set_ylabel('Number of Jets: {}'.format(Jets + MinNoofJets), fontsize = FontSize)

        axes[NoofRows - 1,NoofColumns - 1].set_xlabel('Between {} and {} leptons'.format(MaxNoofLeptons,max(PCADataSet.PRI_nleps)),fontsize = FontSize)
        axes[NoofRows - 1,0].set_ylabel('Between {} and {} jets'.format(MinNoofJets,max(PCADataSet.PRI_jets)), fontsize = FontSize)
        axes1[NoofRows - 1,NoofColumns - 1].set_xlabel('Between {} and {} leptons'.format(MaxNoofLeptons,max(PCADataSet.PRI_nleps)),fontsize = FontSize)
        axes1[NoofRows - 1,0].set_ylabel('Between {} and {} jets'.format(MinNoofJets,max(PCADataSet.PRI_jets)), fontsize = FontSize)
        
    def PCAPlots(self,DataSet,NoofJets,NoofLepton,ax=None,ax1=None,ax2=None, plt_kwargs = {}, sct_kwargs = {}):
        if self.ContainsWeights:
           DataSet2 = DataSet.drop(labels = [self.lbl,'Events_weight'], axis = 1)
        else: 
           DataSet2 = DataSet.drop(labels = self.lbl, axis = 1)
        
        DataSet2.dropna(axis=1,inplace=True)
        scalar = StandardScaler()
        scalar.fit(DataSet2)
        scaled_data =scalar.transform(DataSet2)
        #### Scree plots
        #pca = PCA(n_components= len(DataSet2.columns))
        #pca.fit_transform(DataSet2)
        #plt.plot(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_)
        #plt.title('Scree plot', fontsize = TitleSize)
        #plt.xlabel('Number of PCA components', fontsize = fontsize)
        #plt.ylabel('Eigenvalue', fontsize = fontsize)
        #plt.show()
        ####
        #PCA_Num = click.prompt('From the screen plot how many PCA components would you like to use?', default = 2, type=int)
        pca = PCA(n_components = 2)
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)
        if ax is None:
            ax = plt.gca()
            
        ColourCodes = { 0 : {'Lbl' : 'Background', 'Color' : 'Blue'},
                        1 : {'Lbl' : 'Signal',  'Color' : 'Orange'},
                        'True Negative' : {'Lbl' : 'True Negative', 'Color' : 'Blue'},
                        'False Negative' : {'Lbl' : 'False Negative', 'Color' : 'Orange'},
                        'False positive' : {'Lbl' : 'False positive', 'Color' : 'Green'},
                        'True positive' : {'Lbl' : 'True positive', 'Color' : 'Red'}, 
                        'Signal' : {'Lbl' : 'Signal', 'Color' : 'Orange'},
                        'TTBar' : {'Lbl' : 'TTBar', 'Color' : 'Blue'},
                        'WWBackGround' : {'Lbl' : 'WWChannel', 'Color' : 'Red'}
                      }
        
        Opacity = 1
       
        MeanX = np.mean(x_pca[:,0])
        MeanY = np.mean(x_pca[:,1])
        StdX = np.std(x_pca[:,0])
        StdY = np.std(x_pca[:,1])
        
        if MeanX - (3 * StdX) != MeanX + (3 * StdX):
           ax.set_xlim([MeanX - (3 * StdX),MeanX + (3 * StdX)])
        if MeanY - (3 * StdY) != MeanY + (3 * StdY):
           ax.set_ylim([MeanY - (3 * StdY),MeanY + (3 * StdY)])
        
        for g in np.unique(DataSet[self.lbl]):
            i = np.where(DataSet[self.lbl] == g)
            PercentageOfCase = (len(DataSet[DataSet[self.lbl] == g]) / len(self.df[self.df[self.lbl] == g])) * 100
            
            if self.ContainsWeights:
                SumWeights = sum(DataSet['Events_weight'][DataSet[self.lbl] == g])
                ax.scatter(x_pca[i,0], x_pca[i,1], label = ColourCodes[g]['Lbl'] + ' | ' + str(np.round(PercentageOfCase)) + '%' + '|' + str(round(SumWeights))  , c = ColourCodes[g]['Color'], alpha = Opacity)
            else:
                
                ax.scatter(x_pca[i,0], x_pca[i,1], label = ColourCodes[g]['Lbl'] + ' | ' + str(np.round(PercentageOfCase)) + '%' , c = ColourCodes[g]['Color'], alpha = Opacity)
            #ax.set_title('Principal Component plot of the data for {} jets and {} leptons'.format(NoofJets,NoofLepton))    
            Opacity = Opacity / 2
            
        updatedColumns = list(DataSet2.columns)
        df_comp= pd.DataFrame(pca.components_,columns = updatedColumns)
        if ax1 is None:
            ax1 = plt.gca()
        ax1.set_xlim([-1,1])
        ax1.set_ylim([-1,1])
        ax1.set_xlabel('')
        ax1.scatter(pca.components_[0,:],pca.components_[1,:])
        Circ = plt.Circle([0,0], radius = 1, fill = None)
        ax1.add_patch(Circ)
        ax1.grid()
        txt_height = FontSize / 500
        txt_width = 0.2*(plt.xlim()[1]- plt.xlim()[0])
        text_positions = get_text_positions(pca.components_[0,:],pca.components_[1,:],txt_width,txt_height)
        text_plotter(pca.components_[0,:], pca.components_[1,:], text_positions,updatedColumns , ax1 , txt_width,txt_height)
        #print(dict(zip(updatedColumns,pca.components_[0,:])))
        #print(dict(zip(updatedColumns,pca.components_[1,:])))
        PercentagePCA = abs(np.transpose(pca.components_))/sum(abs(np.transpose(pca.components_)))
        PCAValues = np.transpose(pca.components_)
        #print('Actual values')
        #print(pca.components_)
        #print('Percentage')
        #print(PercentagePCA)
        self.FeaturePCAPercentage['Leptons {} Jets {}'.format(NoofLepton,NoofJets)] = dict(zip(updatedColumns,PercentagePCA))
        self.FeaturePCAValues['Leptons {} Jets {}'.format(NoofLepton,NoofJets)] = dict(zip(updatedColumns,PCAValues))
        if ax2 is None:
           ax2 = plt.gca()
         
        X = np.arange(len(updatedColumns))
        width = 0.25
        PCAOnePercentage = abs(pca.components_[0,:])/sum(abs(pca.components_[0,:]))
        PCATwoPercentage = abs(pca.components_[1,:])/sum(abs(pca.components_[1,:]))
        ax2.bar(X - width/2,PCAOnePercentage, width, color = 'b',label='PCA1')
        ax2.bar(X + width/2,PCATwoPercentage, width, color = 'r',label='PCA2')
        
        ax2.set_ylabel('Percentage of PCA score')
        ax2.set_title('Percentage that each feature makes up of the PCA value', fontsize = TitleSize)
        ax2.set_xticks(X)
        ax2.set_xticklabels([textwrap.fill(label, 30) for label in updatedColumns],rotation='vertical',fontsize=FontSize)
        ax2.legend()
        
        return ax, ax1, ax2
    

        
        


        
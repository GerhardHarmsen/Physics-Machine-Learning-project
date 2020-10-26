# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 19:37:12 2020

@author: gerha
"""
from multiprocessing import Pool, cpu_count
import pandas as pd
import psutil
import numpy as np

def CleanJets(DataSet):
    Tst = pd.DataFrame(columns=DataSet.columns)
    for _,k in DataSet[abs(DataSet.PDGID).isin([1, 2, 3, 4, 5, 6, 7, 8,21])].iterrows():
        DeltaRList = list(np.sqrt((k['DER_Eta'] - DataSet['DER_Eta'][(abs(DataSet.PDGID) == 11)])**2 + (k['DER_Azmithul_Angle'] - DataSet['DER_Azmithul_Angle'][abs(DataSet.PDGID) == 11])**2))
        if len(DeltaRList) > 0:
            if min(DeltaRList) > 0.1:
                Tst = Tst.append(k)
        else:
            Tst = Tst.append(k)
    
    return Tst

def CleanElectrons(DataSet):
    Tst = pd.DataFrame(columns=DataSet.columns)
    for _,k in DataSet[abs(DataSet.PDGID) == 11].iterrows():
        DeltaRList = list(np.sqrt((k['DER_Eta'] - DataSet['DER_Eta'][(abs(DataSet.PDGID).isin([1, 2, 3, 4, 5, 6, 7, 8,21]) )])**2 + (k['DER_Azmithul_Angle'] - DataSet['DER_Azmithul_Angle'][abs(DataSet.PDGID).isin([1, 2, 3, 4, 5, 6, 7, 8,21]) ])**2))
        if len(DeltaRList) > 0:
            if min(DeltaRList) > 0.4:
                Tst = Tst.append(k)
        else:
            Tst = Tst.append(k)
    
    return Tst

if __name__ == '__main__':
    DataSet = pd.read_csv(r'S:\00Gerhard\SampleDataWithDelphes\PsuedoRapidityDataSet.csv')
    DataSet1 = DataSet[(DataSet.IST == 1)]
    Tst = DataSet1[~abs(DataSet1['PDGID']).isin([1, 2, 3, 4, 5, 6, 7, 8,21])]
    n = cpu_count() * 10
    print('Cleaning Data')
    pool = Pool(n)
    JetDataSet = pd.concat(pool.map(CleanJets, [DataSet1[DataSet1.EventID == k] for k in np.unique(DataSet1.EventID)]))
    pool.close()
    pool.join()
    CleanedJetDataSet = pd.concat([Tst,JetDataSet])
    print('Cleaned jets. Now cleaning the electron signals.')
    Tst = CleanedJetDataSet[~abs(CleanedJetDataSet['PDGID']).isin([11])]
    
    pool = Pool(n)    
    ElectronDataSet  = pd.concat(pool.map(CleanElectrons, [CleanedJetDataSet[CleanedJetDataSet.EventID == k] for k in np.unique(CleanedJetDataSet.EventID)]))
    pool.close()
    pool.join()
    print('All events have been cleaned.')
    CleanedDataSet =  pd.concat([Tst,ElectronDataSet])
    
        
            

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:43:06 2020

@author: gerha
"""

import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import uproot
import click
from multiprocessing import Pool, cpu_count

np.seterr(divide='ignore', invalid='ignore')

NoofCPU = cpu_count()
Luminosity = 147
ParticleIDofinterest = [2000013, 1000022]

def CleanJets(Electron, Jet):
    Tst = []
    for k in Jet:
        KeepJet = True
        for l in Electron:
           
            if np.sqrt((k[3] - l[1])**2 + (k[4] - l[2])**2) < 0.1:
                KeepJet = False
        
        if KeepJet: Tst.append(k)
    
    return Tst
            
            
        

def CleanElectrons(Electron, Jet):
    Tst = []
    for k in Electron:
        KeepElect = True
        for l in Jet:
            if np.sqrt((k[1] - l[3])**2 + (k[2] - l[4])**2) < 0.4:
                KeepElect = False
        
        if KeepElect: Tst.append(k)
    
    return Tst

def RemoveColinearProcesses(ElectronDataSet, JetDataSet):
    JetDataSet = CleanJets(ElectronDataSet, JetDataSet)
    ElectronDataSet = CleanElectrons(ElectronDataSet, JetDataSet)
    return JetDataSet, ElectronDataSet

def DivisionTest(Numerator, Denominator):
    try:
        return Numerator/Denominator
    except:
        return np.nan

def CombineEvents(EventData):
    EventID, event_weight, Muon_pt, Muon_eta, Muon_phi, Muon_d0, Muon_dz, Electron_pt, Electron_eta, Electron_phi, Electron_d0, Electron_dz, MET, MET_eta, MET_phi, Tau_Tag, B_Tag, Jet_PT, Jet_Eta, Jet_phi, HTScalar, label_sig = EventData
    if (len(Electron_pt) > 0 ) & (len(Jet_PT) > 0): 
        ElectronSet = list(zip(Electron_pt, Electron_eta, Electron_phi, Electron_d0, Electron_dz))
        JetSet = list(zip(Tau_Tag, B_Tag, Jet_PT, Jet_Eta, Jet_phi))
        JetSet, ElectronSet = RemoveColinearProcesses(ElectronSet, JetSet)
        MuonSet = list(zip(Muon_pt, Muon_eta, Muon_phi, Muon_d0, Muon_dz))
        LeptonSet = ElectronSet + MuonSet
    else: 
        ElectronSet = list(zip(Electron_pt, Electron_eta, Electron_phi, Electron_d0, Electron_dz))
        JetSet = list(zip(Tau_Tag, B_Tag, Jet_PT, Jet_Eta, Jet_phi))
        MuonSet = list(zip(Muon_pt, Muon_eta, Muon_phi, Muon_d0, Muon_dz))
        LeptonSet = ElectronSet + MuonSet
    
    JetSet.sort(key= lambda x: x[2],reverse=True)
    LeptonSet.sort(key= lambda x: x[0],reverse=True)
    EventDataSet = {"EventID" : [],
                    "PRI_nleps" : [],
                    "PRI_jets" : [],
                    "PRI_leading_jet_pt" : [],
                    "PRI_subleading_jet_pt": [],
                    "PRI_leading_jet_eta" : [],
                    "PRI_subleading_jet_eta" : [],
                    "PRI_lep_leading_pt" : [],
                    "PRI_lep_subleading_pt" : [],
                    "PRI_lep_leading_eta" : [],
                    "PRI_lep_subleading_eta" : [],
                    "PRI_lep_leading_phi" : [],
                    "PRI_lep_subleading_phi" : [],
                    "DER_P_T_ratio_lep_pair" : [], 
                    "DER_Diff_Eta_lep_pair" : [],
                    "DER_Diff_Phi_lep_pair" : [],
                    "DER_sum_P_T": [],
                    "PRI_Missing_pt" : [],
                    'HT' : [],
                    'ST' : [],
                    "DER_PT_leading_lepton_ratio_PT_leading_jet": [],
                    "DER_PT_leading_lept_ratio_HT" : [],
                    "DER_ST_ratio_PT_Leading_jet" : [],
                    "DER_ST_ratio_HT" : [],
                    "DER_PT_subleading_lepton_ratio_PT_leading_jet" : [],
                    "DER_PT_subleading_lepton_ratio_HT" : [],
                    "Events_weight" : [],
                    "Label" : []}
    
    EventDataSet["EventID"].append(EventID)
    EventDataSet["Events_weight"].append(event_weight)
    EventDataSet["PRI_jets"].append(len(JetSet))
    if len(JetSet) >= 2:
       EventDataSet["PRI_leading_jet_pt" ].append(JetSet[0][2])
       EventDataSet["PRI_subleading_jet_pt"].append(JetSet[1][2])
       EventDataSet["PRI_leading_jet_eta"].append(JetSet[0][3])
       EventDataSet["PRI_subleading_jet_eta"].append(JetSet[1][3])
    elif len(JetSet) == 1:
       EventDataSet["PRI_leading_jet_pt" ].append(JetSet[0][2])
       EventDataSet["PRI_subleading_jet_pt"].append(np.nan)
       EventDataSet["PRI_leading_jet_eta"].append(JetSet[0][3])
       EventDataSet["PRI_subleading_jet_eta"].append(np.nan)
    elif len(JetSet) == 0:
       EventDataSet["PRI_leading_jet_pt" ].append(np.nan)
       EventDataSet["PRI_subleading_jet_pt"].append(np.nan)
       EventDataSet["PRI_leading_jet_eta"].append(np.nan)
       EventDataSet["PRI_subleading_jet_eta"].append(np.nan) 
           
    EventDataSet["DER_sum_P_T"].append(sum([JetSet[i][2] for i in range(len(JetSet))] + [LeptonSet[i][0] for i in range(len(LeptonSet))]))
    ### Determing the values associated to the leading and sub-leading leptons###
    EventDataSet["PRI_nleps"].append(len(LeptonSet))
    if len(LeptonSet) >= 2:
            ###Values for leading and sub-leading leptons######
            EventDataSet["PRI_lep_leading_pt"].append(LeptonSet[0][0])
            EventDataSet["PRI_lep_subleading_pt"].append(LeptonSet[1][0])
            
            EventDataSet["PRI_lep_leading_eta"].append(LeptonSet[0][1])
            EventDataSet["PRI_lep_subleading_eta"].append(LeptonSet[1][1])
            
            EventDataSet["PRI_lep_leading_phi"].append(LeptonSet[0][2])
            EventDataSet["PRI_lep_subleading_phi"].append(LeptonSet[1][2])
            ###Comparisons between leading and sub-leading leptons#####
            EventDataSet["DER_P_T_ratio_lep_pair"].append(LeptonSet[0][0]/LeptonSet[1][0])
            EventDataSet["DER_Diff_Eta_lep_pair"].append(abs(LeptonSet[0][1] - LeptonSet[1][1] ))
            EventDataSet["DER_Diff_Phi_lep_pair"].append(abs(LeptonSet[0][2] - LeptonSet[1][2] ))
            
            
    elif len(LeptonSet) == 1:
            ###Values for leading and sub-leading leptons######
            EventDataSet["PRI_lep_leading_pt"].append(LeptonSet[0][0])
            EventDataSet["PRI_lep_subleading_pt"].append(np.nan)
            
            EventDataSet["PRI_lep_leading_eta"].append(LeptonSet[0][1])
            EventDataSet["PRI_lep_subleading_eta"].append(np.nan)
            
            EventDataSet["PRI_lep_leading_phi"].append(LeptonSet[0][2])
            EventDataSet["PRI_lep_subleading_phi"].append(np.nan)
            ###Comparisons between leading and sub-leading leptons#####
            EventDataSet["DER_P_T_ratio_lep_pair"].append(np.nan)
            EventDataSet["DER_Diff_Eta_lep_pair"].append(np.nan)
            EventDataSet["DER_Diff_Phi_lep_pair"].append(np.nan)
            
            
    elif len(LeptonSet) == 0:
            ###Values for leading and sub-leading leptons######
            EventDataSet["PRI_lep_leading_pt"].append(np.nan)
            EventDataSet["PRI_lep_subleading_pt"].append(np.nan)
            
            EventDataSet["PRI_lep_leading_eta"].append(np.nan)
            EventDataSet["PRI_lep_subleading_eta"].append(np.nan)
            
            EventDataSet["PRI_lep_leading_phi"].append(np.nan)
            EventDataSet["PRI_lep_subleading_phi"].append(np.nan)
            ###Comparisons between leading and sub-leading leptons#####
            EventDataSet["DER_P_T_ratio_lep_pair"].append(np.nan)
            EventDataSet["DER_Diff_Eta_lep_pair"].append(np.nan)
            EventDataSet["DER_Diff_Phi_lep_pair"].append(np.nan)

        ###Missing Energy values#####
    EventDataSet["PRI_Missing_pt"].append(sum(MET))
    
    EventDataSet['Label'].append(label_sig)
    
    #### Added features from paper 1901.XXX
        
    ST = EventDataSet["PRI_lep_leading_pt"][0] + EventDataSet["PRI_lep_subleading_pt"][0]
    EventDataSet['HT'].append(HTScalar[0])
    EventDataSet['ST'].append(ST)
    EventDataSet["DER_PT_leading_lepton_ratio_PT_leading_jet"].append(DivisionTest(EventDataSet["PRI_lep_leading_pt"][0],EventDataSet["PRI_leading_jet_pt" ][0]))
    EventDataSet["DER_PT_leading_lept_ratio_HT"].append(DivisionTest(EventDataSet["PRI_lep_leading_pt"][0],HTScalar[0]))
    EventDataSet["DER_ST_ratio_PT_Leading_jet"].append(DivisionTest(ST,EventDataSet["PRI_leading_jet_pt" ][0])) 
    EventDataSet["DER_ST_ratio_HT"].append(DivisionTest(ST,HTScalar[0]))
    EventDataSet["DER_PT_subleading_lepton_ratio_PT_leading_jet"].append(DivisionTest(EventDataSet["PRI_lep_subleading_pt"][0],EventDataSet["PRI_lep_leading_pt"][0]))
    EventDataSet["DER_PT_subleading_lepton_ratio_HT"].append(DivisionTest(EventDataSet["PRI_lep_subleading_pt"][0],HTScalar[0])) 
    
    return  pd.DataFrame(EventDataSet)

def EventWeight(ROOTFILE):
    head, tail = os.path.split(ROOTFILE)
    head, tail = os.path.split(head)
    
    try:
        File = open(os.path.join(head,'run_01_banner.txt'), 'r')
        LineText = File.readlines()
        del LineText[0:LineText.index("<MGGenerationInfo>\n") + 1 ]
        NoofEvents = int(LineText[0].strip().split()[-1])
        IntegratedWeight = float(LineText[1].strip().split()[-1])
        return (IntegratedWeight/NoofEvents) * 147 * 1000
        File.close()
    except:
        print(head)
        print('Cannot find file containing the weights of the events.')
        sys.exit('Stopping program')
    

def DelphesFile(ROOTFILE, EventID):
    File = uproot.open(ROOTFILE)
    # Read data from ROOT files
    TREE = File['Delphes']
    BRANCH = TREE['Event']
    NoofEvents = len(BRANCH['Event.Weight'].array())
    BRANCH = TREE['Weight']
    event_weight = EventWeight(ROOTFILE)
    event_weight = [event_weight] * NoofEvents
    #event_weight =  BRANCH['Weight.Weight'].array() * Luminosity * 1000
    LocEventID = list(range(EventID, EventID + NoofEvents))
    
    BRANCH = TREE['Particle']
    PDGID = BRANCH['Particle.PID'].array()
    
    
    if any(item in abs(PDGID[0]) for item in ParticleIDofinterest):
        label_sig = [1] * NoofEvents
    else: 
        label_sig = [0] * NoofEvents
    

    BRANCH = TREE['Muon']
    Muon_pt = BRANCH['Muon.PT'].array()
    Muon_eta = BRANCH['Muon.Eta'].array()
    Muon_phi = BRANCH['Muon.Phi'].array()
    Muon_d0 = BRANCH['Muon.D0'].array()
    Muon_dz = BRANCH['Muon.DZ'].array()
    
    BRANCH = TREE['Electron']
    Electron_pt = BRANCH['Electron.PT'].array()
    Electron_eta = BRANCH['Electron.Eta'].array()
    Electron_phi = BRANCH['Electron.Phi'].array()  
    Electron_d0 = BRANCH['Electron.D0'].array()
    Electron_dz = BRANCH['Electron.DZ'].array()
    
    BRANCH = TREE['MissingET']
    MET = BRANCH['MissingET.MET'].array()
    MET_eta = BRANCH['MissingET.Eta'].array()
    MET_phi = BRANCH['MissingET.Phi'].array()

    BRANCH = TREE['Jet']
    Tau_Tag = BRANCH['Jet.TauTag'].array()
    B_Tag = BRANCH['Jet.BTag'].array()
    Jet_PT = BRANCH['Jet.PT'].array()
    Jet_Eta = BRANCH['Jet.Eta'].array()
    Jet_phi = BRANCH['Jet.Phi'].array()
    
    BRANCH = TREE['ScalarHT']
    HTScalar = BRANCH['ScalarHT.HT'].array()
    
    ZippedData =  list(zip(LocEventID,
                      event_weight,
                      Muon_pt,
                      Muon_eta,
                      Muon_phi,
                      Muon_d0,
                      Muon_dz, 
                      Electron_pt,
                      Electron_eta,
                      Electron_phi,
                      Electron_d0,
                      Electron_dz,
                      MET,
                      MET_eta,
                      MET_phi,
                      Tau_Tag,
                      B_Tag,
                      Jet_PT,
                      Jet_Eta,
                      Jet_phi,
                      HTScalar,
                      label_sig))
    DataSet = CombineEvents(ZippedData[0])
    with Pool(NoofCPU) as pool:
         DataSet = pd.concat(pool.imap(CombineEvents,[Event for Event in ZippedData]))
    
    pool.close()
    pool.join()
    return DataSet

def DELPHESTOCSV2(SelectedDirectory = None, OutputDirectory = None):
    if SelectedDirectory == None:
        sys.exit('No directory chosen. Please select the directory containing the events.')
    try:
        SelectedDirectory = os.path.normpath(SelectedDirectory)
        
    except:
        sys.exit('Unable to find selected directory.')
        
        
    if OutputDirectory == None:
        print('Files will be saved in {}'.format(str(SelectedDirectory)))
        OutputDirectory = SelectedDirectory
    else:
        try:
            OutputDirectory = os.path.normpath(OutputDirectory)
        except:
            sys.exit('Unable to find output directory.') 
        
        if os.path.exists(OutputDirectory):
           print('Output folder found.')
        else:
           sys.exit('Unable to find output directory.')   
            
            
    
    FileList = []
    EventID = 1
    for root, dirs, files in os.walk(SelectedDirectory):
        for names in files:
             if "tag_1_delphes_events.root" in os.path.join(root, names):
                FileList.append( os.path.join(root, names))
                  
    for Files in tqdm(FileList):
        if EventID == 1:
            DataFrame = DelphesFile(Files, EventID)
            EventID = max(DataFrame.EventID) + 1
        else:
            DataFrame = DataFrame.append(DelphesFile(Files, EventID))
            EventID = max(DataFrame.EventID) + 1
    print('Found {} events'.format(max(DataFrame.EventID)))
    print('Finished converting. Saving file....')
    print(DataFrame.head()) 
    DataFrame.to_csv(os.path.join(OutputDirectory,"EventData.csv"), index = False)    

def DELPHESTOCSV(SelectedDirectory):
    
    EventID = 1
    SelectedDirectory = os.path.normpath(SelectedDirectory)
    for root, dirs, files in os.walk(SelectedDirectory):
        for names in files:
             if "tag_1_delphes_events.root" in os.path.join(root, names):
                 if EventID == 1:
                     DataFrame = DelphesFile(os.path.join(root, names), EventID)
                     EventID = max(DataFrame.EventID) + 1
                 else:
                     DataFrame = DataFrame.append(DelphesFile(os.path.join(root, names), EventID))
                     EventID = max(DataFrame.EventID) + 1
    print('Found {} events'.format(max(DataFrame.EventID)))
    print('Finished converting. Saving file....')
    print(DataFrame.head()) 
    DataFrame.to_csv(SelectedDirectory + r"\EventData.csv", index = False)    

def NoofDelphesFiles(SelectedDirectory):
    """Tests if there are ROOT files in the directory. 
    
    Parameters
    ----------
    SelectedDirectory : Path
        Path to the desired folder in which to check for LHE files. Note: The function will check subdirectories as well.

    Returns
    -------
    None.
    """
    
    print('Checking for ROOT files')
    i = 0 
    FileList = []
    for root, dirs, files in os.walk(SelectedDirectory):
        if any("tag_1_delphes_events.root" in s for s in files):
            
            i = i + 1
    if i > 0:
      print("{} ROOT files found".format(i ))
      if click.confirm('Would you like to convert the ROOT files?'):
         print('Converting files in directory {}'.format(SelectedDirectory))
      else:
        sys.exit('Stopping program')  
    else:
      sys.exit('No files found ending execution')


if __name__ == '__main__':
   from tkinter import filedialog as fd
   
   SelectedDirectory = fd.askdirectory() 
   NoofDelphesFiles(SelectedDirectory)
   DELPHESTOCSV2(SelectedDirectory)

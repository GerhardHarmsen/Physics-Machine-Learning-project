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
import vector
from mt2 import mt2

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
    
def MuonMT2(Muon1,Muon2, MET, MET_eta, MET_phi, InvisibleMass):
    MissingEnergy = vector.obj(rho=MET[0],eta=MET_eta[0],phi=MET_phi[0])
    
    val = mt2(
    0.11, Muon1.x, Muon1.y,  # Visible 1: mass, px, py
    0.11, Muon2.x, Muon2.y,  # Visible 2: mass, px, py
    MissingEnergy.x, MissingEnergy.y,  # Missing transverse momentum: x, y
    0, 0)  # Invisible 1 mass, invisible 2 mass
    return val

def CombineEvents(EventData):
    EventID, event_weight, Muon_pt, Muon_eta, Muon_phi, Muon_d0, Muon_dz, Electron_pt, Electron_eta, Electron_phi, Electron_d0, Electron_dz, MET, MET_eta, MET_phi, InvisibleMass, Tau_Tag, B_Tag, Jet_PT, Jet_Eta, Jet_phi, HTScalar, label_sig = EventData
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
                    "PRI_nMuons" : [],
                    "PRI_Muon_leading_pt" : [],
                    "PRI_Muon_subleading_pt" : [],
                    "PRI_Muon_leading_eta" : [],
                    "PRI_Muon_subleading_eta" : [],
                    "PRI_Muon_leading_phi" : [],
                    "PRI_Muon_subleading_phi" : [],
                    "DER_Muon_invariant_mass" : [],
                    "DER_MT2_variable" : [],
                    "Events_weight" : [],
                    "Label" : []}
    
    EventDataSet["EventID"].append(EventID)
    EventDataSet["Events_weight"].append(event_weight)
    ### JET INFORMATION ###############################
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
    ########## MUON DATA ############################################
    EventDataSet["PRI_nMuons"].append(len(MuonSet))
    if len(MuonSet) >= 2:
            ###Values for leading and sub-leading muons######
            EventDataSet["PRI_Muon_leading_pt"].append(MuonSet[0][0])
            EventDataSet["PRI_Muon_subleading_pt"].append(MuonSet[1][0])
            
            EventDataSet["PRI_Muon_leading_eta"].append(MuonSet[0][1])
            EventDataSet["PRI_Muon_subleading_eta"].append(MuonSet[1][1])
            
            EventDataSet["PRI_Muon_leading_phi"].append(MuonSet[0][2])
            EventDataSet["PRI_Muon_subleading_phi"].append(MuonSet[1][2])
            ###GET THE MUON INVARIANT MASS####
            Muon1 = vector.obj(rho=MuonSet[0][0],eta=MuonSet[0][1],phi=MuonSet[0][2],mass=0.11)
            Muon2 = vector.obj(rho=MuonSet[1][0],eta=MuonSet[1][1],phi=MuonSet[1][2],mass=0.11)
            InvMass = (Muon1+Muon2).mass
            EventDataSet["DER_Muon_invariant_mass"].append(InvMass)
            EventDataSet["DER_MT2_variable"].append(MuonMT2(Muon1,Muon2, MET, MET_eta, MET_phi, InvisibleMass))
                
            
            
    elif len(MuonSet) == 1:
            ###Values for leading and sub-leading muons######
            EventDataSet["PRI_Muon_leading_pt"].append(MuonSet[0][0])
            EventDataSet["PRI_Muon_subleading_pt"].append(np.nan)
            
            EventDataSet["PRI_Muon_leading_eta"].append(MuonSet[0][1])
            EventDataSet["PRI_Muon_subleading_eta"].append(np.nan)
            
            EventDataSet["PRI_Muon_leading_phi"].append(MuonSet[0][2])
            EventDataSet["PRI_Muon_subleading_phi"].append(np.nan)
            ###GET THE MUON INVARIANT MASS####
            EventDataSet["DER_Muon_invariant_mass"].append(0.11)
            EventDataSet["DER_MT2_variable"].append(np.nan)
            
            
    elif len(MuonSet) == 0:
            ###Values for leading and sub-leading muons######
            EventDataSet["PRI_Muon_leading_pt"].append(np.nan)
            EventDataSet["PRI_Muon_subleading_pt"].append(np.nan)
            
            EventDataSet["PRI_Muon_leading_eta"].append(np.nan)
            EventDataSet["PRI_Muon_subleading_eta"].append(np.nan)
            
            EventDataSet["PRI_Muon_leading_phi"].append(np.nan)
            EventDataSet["PRI_Muon_subleading_phi"].append(np.nan)
            ###GET THE MUON INVARIANT MASS####
            EventDataSet["DER_Muon_invariant_mass"].append(np.nan)
            EventDataSet["DER_MT2_variable"].append(np.nan)
    
    
    
    return  pd.DataFrame(EventDataSet)


def DetectMADSpinRun(ROOTFILE_Event_folder):
    ROOTFILE_Event_folder = os.path.join(ROOTFILE_Event_folder,'run_01')
    
    try:
        with open(os.path.join(ROOTFILE_Event_folder,'MADSpin_Check.txt'),'r') as MyFile:
            LineText = MyFile.readlines()
            del LineText[0:LineText.index("<MADSpinTest>\n") + 1 ]
            if LineText[0].strip().split()[-1] == 'True':
                return True
            else:
                return False
            
    except:
        with uproot.open(os.path.join(ROOTFILE_Event_folder,'unweighted_events.root')) as RootFile:
            TREE = RootFile['LHEF']
            BRANCH = TREE['Event']
            Nparticles = list(BRANCH['Event.Nparticles'].array())
        Results = np.unique(Nparticles)
        Diff = [Results[i] - Results[i-1] for i in range(1,len(Results))]
        with open(os.path.join(ROOTFILE_Event_folder,'MADSpin_Check.txt'),'w') as MyFile:
            MyFile.write("When MADSpin is run the number of deteted events in the root files is not reliable, as it doubles the number of events.\n")
            MyFile.write("This file tells the script DelphesToCSV.py written by Gerhard Harmsen whether or not MADSpin has been written.\n")
            MyFile.write('The script will then know to divide the number of events given in "run_banner.txt" by 2 to get the correct number of events.\n')
            MyFile.write('This allows for the correct event weights when MADSpin has been run.\n')
            MyFile.write('<MADSpinTest>\n')
            if max(Diff) > 1:
                MyFile.write('MADSpin run : True\n')
            else:
                MyFile.write('MADSpin run : False\n')
            MyFile.write('</MADSpinTest>\n')
        
        if max(Diff) > 1:
            return True
        else:
            return False


def EventWeight(ROOTFILE):
    head, _ = os.path.split(ROOTFILE)
    head, _ = os.path.split(head)
    with open(os.path.join(head,'run_01_banner.txt'), 'r') as MGFile:
        LineText = MGFile.readlines()
    del LineText[0:LineText.index("<MGGenerationInfo>\n") + 1 ]
    cross_section = float(LineText[1].strip().split()[-1])
    NoofEvents = int(LineText[0].strip().split()[-1])
   
    MADSpin =  DetectMADSpinRun(head)       
    if MADSpin:
        NoofEvents = NoofEvents / 2
    
    return ((cross_section/NoofEvents) * 147 * 1000)
        
        
def NeutralinoMass(ROOTFILE):
    head, tail = os.path.split(ROOTFILE)
    head, tail = os.path.split(head)
    
    File = open(os.path.join(head,'run_01_banner.txt'), 'r')
    LineText = File.readlines()
    NeutrinoMass = 0
    for i in LineText:
        if '1000022' in i :
           if 'mneu1' in i:
              NeutrinoMass = float(i.strip().split()[1])
              break
    return NeutrinoMass
    

def DelphesFile(ROOTFILE, EventID, DataSet_Label):
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
    
    
    label_sig = [DataSet_Label] * NoofEvents
    

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
    
    InvisibleMass = [NeutralinoMass(ROOTFILE)] * NoofEvents
    
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
                      InvisibleMass,
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

def DELPHESTOCSV2(DataSet_Label, SelectedDirectory = None, OutputDirectory = None):
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
            DataFrame = DelphesFile(Files, EventID, DataSet_Label)
            EventID = max(DataFrame.EventID) + 1
        else:
            DataFrame = DataFrame.append(DelphesFile(Files, EventID, DataSet_Label))
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
                 print(os.path.join(root, names))
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
   DataSet_Label = input('Type 1 for signal and 0 for background type events.')
   NoofDelphesFiles(SelectedDirectory)
   DELPHESTOCSV2(DataSet_Label, SelectedDirectory) 
   print()

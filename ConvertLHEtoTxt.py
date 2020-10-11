# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:08:46 2020

@author: Gerhard Erwin Harmsen
"""
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from tkinter import filedialog as fd
import multiprocessing as mp
import uproot
import click
from multiprocessing import Pool, cpu_count

NoofCPU = mp.cpu_count()
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
#print("Files in %r: %s" % (cwd, files))

Luminosity = 147 #Luminosity of the LHC as of August 2020

def TESTFORLHE(SelectedDirectory):
    """Function for testing if there are LHE files in the selected directory. 
    
    Parameters
    ----------
    SelectedDirectory : Path
        Path to the desired folder in which to check for LHE files. Note: The function will check subdirectories as well.

    Returns
    -------
    True if there are LHE files and false otherwise.
    """
    
    i = 0 
    for root, dirs, files in os.walk(SelectedDirectory):
        for name in files:
            if "unweighted_events.lhe" in os.path.join(root, name):
              if "unweighted_events.lhe.gz" not in os.path.join(root, name):   
                 i = i + 1
    assert(i != 0 ), "No data files. Ensure that you have selected a folder containing the files 'unweighted_events.lhe'."
    print("Found {} LHE unweighted event files.\n".format(i))
    
def ROOTFILES(SelectedDirectory):
    """Tests if there are ROOT files in the directory. 
    
    Parameters
    ----------
    SelectedDirectory : Path
        Path to the desired folder in which to check for LHE files. Note: The function will check subdirectories as well.

    Returns
    -------
    If there are none or the users chooses to use LHE files then it reurns false.
    """
    
    print('Checking for ROOT files')
    i = 0 
    for root, dirs, files in os.walk(SelectedDirectory):
        if any(".root" in s for s in files):
            i = i + 1
    if i > 0:
      print("{} ROOT files found".format(i ))
      if click.confirm('Would you like to convert the ROOT files?'):
         return True
      else:
         TESTFORLHE(SelectedDirectory)
         return False 
    else:
      TESTFORLHE(SelectedDirectory)
      return False
           
def ExtractInfo(String):
    """Returns the particle specific values for a particular particle in a LHE event file. 
    
    Parameters
    ----------
    String : String
        The string must be a string in event section of the LHE file, but must not be the first line in the section.

    Returns
    -------
    All features in the LHE string.
    """
    String.strip()
    String = String.split()
    ID = String[0]          #   PDG code                                        
    IST = String[1]         #   status code
    MOTH1 = String[2]       #   row number corresponding to the first mother particle 
    MOTH2 = String[3]       #   row number corresponding to the second mother particle
    ICOL1 = String[4]       #   first color tag  
    ICOL2 = String[5]       #   second color tag
    P1 = String[6]          #   PX 
    P2 = String[7]          #   PY
    P3 = String[8]          #   PZ  
    P4 = String[9]          #   E         
    P5 = String[10]         #   M (a space-like virtuality is denoted by a negative mass)
    VTIM = String[11]       #   c tau 
    SPIN = String[12]       #   cosine of the angle between the spin vector of the particle and its three-momentum  
    return([ID, IST, MOTH1, MOTH2, ICOL1, ICOL2, P1, P2, P3, P4, P5, VTIM, SPIN]) 
    
def ValueP_T(P1, P2):
    """Returns the transfers momentum of the the particle. 
    
    Parameters
    ----------
    P1: Float
    The x-momentum of the particle.
    
    P2: Float
    The y-momentum of the particle.
    Returns
    -------
    Transferse momentum of the particle.
    """
    return np.sqrt(P1**2+P2**2)

def ValueEta(P1, P2, P3):
    """Returns the pseudorapidity of the particle.
    
    Parameters:
       
    
    Parameters
    ----------
     P1: Float 
     The x-momentum of the particle.
     
     P2: Float 
     The y-momentum of the particle.
     
     P3: Float
     The z-momentum of the particle. 
    Returns
    -------
    Pseudorapidity value of the particle.
    """
    
    CosTheta = P3/np.sqrt(P1**2  + P2**2 + P3**2)
    return -0.5*np.log((1-CosTheta)/(1+CosTheta))

def ValueAzmithulAngle(P1, P2, P3, P_T):
    """ Returns the Azmithul angle of the particle.
   
    Parameters
    ----------
    P1: Float
    The x-momentum of the particle.
    
    P2: Float
    The y-momentum of the particle.
    
    P3: Float
    The z-momentum of the particle. 
    
    P_T: FLoat
    The transfers momentum of the particle.

    Returns
    -------
    Azimithul angle of the particle.        
            """
    return np.arcsin(P_T/np.sqrt(P1**2  + P2**2 + P3**2))

def CreateFile(Folder_selected):
    """
    Creates a csv file with the information from an LHE file

    Parameters
    ----------
    Folder_selected : Path to folder containing LHE files, or subdirectories containing LHE files.
        The path is where the csv file will be saved. It must also either cointain LHE files or subfolders that contain LHE files.

    Returns
    -------
    None

    """
    NewFile = open(Folder_selected + r'\LHEEventData.csv','w+', newline='')
    writer = csv.writer(NewFile)
    writer.writerow(["EventID","PDGID","IST","MOTH1","MOTH2","ICOL1","ICOL2","P1","P2","P3","P4","P5","VTIM","SPIN","WEIGHT"])
    EventID = 1
    
    for root, dirs, files in os.walk(Folder_selected):
      for name in files:
         if "unweighted_events.lhe" in os.path.join(root, name):
             if "unweighted_events.lhe.gz" not in os.path.join(root, name):
                 NewEventID , WList = ConverttoText(os.path.join(root, name),EventID)
                 EventID = NewEventID
                 for item in WList:
                     writer.writerow(item)
                     
    print("Number of events detected: {}".format(EventID -1))                 
    NewFile.close()


def ConverttoText(LHEFile, EventID):
    """
    Converts the information of a particular LHE File to a list of event data and a assigns eventIDs to the events, such that they can be written to a file.

    Parameters
    ----------
    LHEFile : String
        Path to the LHE file.
        
    EventID : Integer
        Value of the first event ID to be used. 

    Returns
    -------
    list
        LocEventID is the largest value for  the eventID used.
        The List of events and all relevant information about the event stored in the LHE file. 

    """
    File = open(LHEFile, 'r')
    LineText = File.readlines()
    LineText = [i.strip() for i in LineText]
    del LineText[0:LineText.index("</init>") + 1 ]
    EventWeights = np.float(LineText[LineText.index('<event>') + 1].split()[2]) * Luminosity * 1000
    WriteList = []
    LocEventID = EventID
    pbar = tqdm(total = len(LineText), leave = None)
    
    while len(LineText) !=1 :
        try:
            for x in LineText[LineText.index('<event>')+2:LineText.index('<mgrwt>')]:
               if "scales" not in x:
                   WriteList.append([LocEventID] + ExtractInfo(x) + [EventWeights])
        except:
            for x in LineText[LineText.index('<event>')+2:LineText.index('</event>')]:
                if "scales" not in x :
                    WriteList.append([LocEventID] + ExtractInfo(x) + [EventWeights])
        try:
            pbar.update(LineText.index('</event>') + 1 - LineText.index('<event>'))
        except:
            print(LineText)
        del LineText[LineText.index('<event>'):LineText.index('</event>') +1] 
        LocEventID = LocEventID + 1   
            
    pbar.leave = None
    pbar.update(len(LineText))   
    pbar.close()   
    File.close()
    return [LocEventID, WriteList]


def ConvertoPseudorapidity(SelectedDirectory):
    """
    Creates a file that contains information about the pseudorapidity of the particle events in a .csv file.

    Parameters
    ----------
    SelectedDirectory : String
        Path containing the LHEEventData.csv file to allow for converting to psuedorapitiy data.

    Returns
    -------
    None.

    """
    DataSet = pd.read_csv(SelectedDirectory + r'\LHEEventData.csv' )
    print(DataSet.head())
    DataSet['DER_P_T'] = ValueP_T(DataSet.P1, DataSet.P2)
    DataSet['DER_Eta'] = ValueEta(DataSet.P1,DataSet.P2, DataSet.P3)
    DataSet['DER_Azmithul_Angle'] = ValueAzmithulAngle(DataSet.P1, DataSet.P2, DataSet.P3, DataSet.DER_P_T)
    print('Finished adding pseudorapidities. Saving file....')
    DataSet.to_csv(SelectedDirectory + r"\PsuedoRapidityDataSet.csv", index = False)
    print(DataSet.head())   



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

def RemoveColinearEvents(DataSet):
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
    return CleanedDataSet
        
def CombineEvent(DataSet)       
    
def RecombineEvents(SelectedDirectory):
    """
    Comines all the particle decays in a particular event together. It then creates a csv file in the Selected Directory with all the combined events.
    
    Parameters
    ----------
    SelectedDirectory : String
        Path of directory containing PsuedoRapidityDataSet file, and directory to which the combined events will be saved.

    Returns
    -------
    None.

    """
    PDGID_Lepton_List = [11, 13, 15, 17]
    PDGID_Quark_List =  [1, 2, 3, 4, 5, 6, 7, 8]
    PDGID_Boson_List = [9, 21, 22, 23, 24, 25, 37]
    PDGID_Neutrino_List = [12, 14, 16, 18, -1000022, 1000022]
    ParticleIDofinterest = [-1000022, 1000022]
    DataSet = pd.read_csv(SelectedDirectory + r"\PsuedoRapidityDataSet.csv")
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
                    "Events_weight" : [],
                    "Label" : []}
    DataSet = RemoveColinearEvents(DataSet)
    for i in tqdm(np.unique(DataSet.EventID), leave = None):   
        Temp = DataSet[DataSet.EventID == i]
        EventDataSet["EventID"].append(i)
        EventDataSet["Events_weight"].append(np.unique(DataSet['WEIGHT'][DataSet.EventID == i])[0])
        EventDataSet["PRI_jets"].append(len(Temp[(Temp.IST == 1) & (abs(Temp.PDGID).isin(PDGID_Quark_List + [21]))]))
        Tst = Temp[(Temp.IST == 1) & (abs(Temp.PDGID).isin(PDGID_Quark_List + [21]))]
        Tst = Tst.sort_values('DER_P_T', ascending = False)
        if len(Tst) >= 2:
           EventDataSet["PRI_leading_jet_pt" ].append(Tst["DER_P_T"].iloc[0])
           EventDataSet["PRI_subleading_jet_pt"].append(Tst["DER_P_T"].iloc[1])
           EventDataSet["PRI_leading_jet_eta"].append(Tst['DER_Eta'].iloc[0])
           EventDataSet["PRI_subleading_jet_eta"].append(Tst['DER_Eta'].iloc[1])
        elif len(Tst) == 1:
           EventDataSet["PRI_leading_jet_pt" ].append(Tst["DER_P_T"].iloc[0])
           EventDataSet["PRI_subleading_jet_pt"].append(np.nan)
           EventDataSet["PRI_leading_jet_eta"].append(Tst['DER_Eta'].iloc[0])
           EventDataSet["PRI_subleading_jet_eta"].append(np.nan)
        elif len(Tst) == 0:
           EventDataSet["PRI_leading_jet_pt" ].append(np.nan)
           EventDataSet["PRI_subleading_jet_pt"].append(np.nan)
           EventDataSet["PRI_leading_jet_eta"].append(np.nan)
           EventDataSet["PRI_subleading_jet_eta"].append(np.nan) 
           
           
        EventDataSet["DER_sum_P_T"].append(sum(Temp.DER_P_T[Temp.IST ==  1]))
        ### Determing the values associated to the leading and sub-leading leptons###
        Tst = Temp[(Temp.IST == 1) & (abs(Temp.PDGID).isin(PDGID_Lepton_List))]
        Tst = Tst.sort_values('DER_P_T', ascending = False)
        EventDataSet["PRI_nleps"].append(len(Tst))

        if len(Tst) >= 2:
            ###Values for leading and sub-leading leptons######
            EventDataSet["PRI_lep_leading_pt"].append(Tst['DER_P_T'].iloc[0])
            EventDataSet["PRI_lep_subleading_pt"].append(Tst['DER_P_T'].iloc[1])
            
            EventDataSet["PRI_lep_leading_eta"].append(Tst['DER_Eta'].iloc[0])
            EventDataSet["PRI_lep_subleading_eta"].append(Tst['DER_Eta'].iloc[1])
            
            EventDataSet["PRI_lep_leading_phi"].append(Tst['DER_Azmithul_Angle'].iloc[0])
            EventDataSet["PRI_lep_subleading_phi"].append(Tst['DER_Azmithul_Angle'].iloc[1])
            ###Comparisons between leading and sub-leading leptons#####
            EventDataSet["DER_P_T_ratio_lep_pair"].append(Tst['DER_P_T'].iloc[0]/Tst['DER_P_T'].iloc[1])
            EventDataSet["DER_Diff_Eta_lep_pair"].append(abs(Tst['DER_Eta'].iloc[0] - Tst['DER_Eta'].iloc[1]))
            EventDataSet["DER_Diff_Phi_lep_pair"].append(abs(Tst['DER_Azmithul_Angle'].iloc[0] - Tst['DER_Azmithul_Angle'].iloc[1]))
            
            
        elif len(Tst) == 1:
            ###Values for leading and sub-leading leptons######
            EventDataSet["PRI_lep_leading_pt"].append(Tst['DER_P_T'].iloc[0])
            EventDataSet["PRI_lep_subleading_pt"].append(np.nan)
            
            EventDataSet["PRI_lep_leading_eta"].append(Tst['DER_Eta'].iloc[0])
            EventDataSet["PRI_lep_subleading_eta"].append(np.nan)
            
            EventDataSet["PRI_lep_leading_phi"].append(Tst['DER_Azmithul_Angle'].iloc[0])
            EventDataSet["PRI_lep_subleading_phi"].append(np.nan)
            ###Comparisons between leading and sub-leading leptons#####
            EventDataSet["DER_P_T_ratio_lep_pair"].append(np.nan)
            EventDataSet["DER_Diff_Eta_lep_pair"].append(np.nan)
            EventDataSet["DER_Diff_Phi_lep_pair"].append(np.nan)
            
            
        elif len(Tst) == 0:
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
        EventDataSet["PRI_Missing_pt"].append(sum(Temp.DER_P_T[(Temp.IST == 1) & (Temp.PDGID.isin(PDGID_Neutrino_List))]))
        
        if any(item in list(Temp.PDGID[Temp.IST == 1]) for item in ParticleIDofinterest):
            EventDataSet['Label'].append(1)
        else:
             EventDataSet['Label'].append(0)
       
    
    df = pd.DataFrame(EventDataSet)
    print("Sample of the data.")
    print(df.head())
    df.to_csv(SelectedDirectory + "\EventData.csv", index = False)    
    
def ExtractFromROOT(SelectedFile, EventID):
    """
    Extracts information from a ROOT file and captures all relevant event data, including psuedorapidities.    

    Parameters
    ----------
    SelectedFile : String
        Path to desired ROOT file.
    EventID : Integer
        Starting Event ID to be written to the dataset

    Returns
    -------
    DataSet : Pandas DataSet
        The dataset containing all information from the converted ROOT file.

    """
    ROOTFILE = uproot.open(SelectedFile)
    LocEventID = EventID
    TREE = ROOTFILE['LHEF']
    BRANCH = TREE['Event']
    EventWeights = np.float(BRANCH['Event.Weight'].array()[0][0]) * Luminosity * 1000 ### WEIGHT
    BRANCH = TREE['Particle']
    WriteList = []
    ID = BRANCH['Particle.PID'].array() ### PDGID
    IST = BRANCH['Particle.Status'].array() ### IST
    MOTH1 = BRANCH['Particle.Mother1'].array() ### MOTH1
    MOTH2 = BRANCH['Particle.Mother2'].array() ### MOTH2
    ICOL1 = BRANCH['Particle.ColorLine1'].array() ### ICOL1
    ICOL2 = BRANCH['Particle.ColorLine2'].array() ### ICOL2
    P1 = BRANCH['Particle.Px'].array() ### P1
    P2 = BRANCH['Particle.Py'].array() ### P2
    P3 = BRANCH['Particle.Pz'].array() ### P3
    P4 = BRANCH['Particle.E'].array() ### P4
    P5 = BRANCH['Particle.M'].array() ### P5 
    VTIM = BRANCH['Particle.LifeTime'].array() ### VTIM
    SPIN = BRANCH['Particle.Spin'].array() ### SPIN 
    DER_P_T = BRANCH['Particle.PT'].array() ### DER_P_T
    DER_Eta = BRANCH['Particle.Eta'].array()### DER_Eta
    DER_Azmithul_Angle =  BRANCH['Particle.Phi'].array() ### DER_Azmithul_Angle    
    pbar = tqdm(total = len(BRANCH['Particle.fUniqueID'].array()) // 2, leave = None)
    for i in range(len(BRANCH['Particle.fUniqueID'].array()) // 2, len(BRANCH['Particle.fUniqueID'].array()) ):
         WriteList = WriteList + list(zip([LocEventID for k in range(len(ID[i]))], ID[i], IST[i], MOTH1[i], MOTH2[i], ICOL1[i], ICOL2[i], P1[i], P2[i], P3[i], P4[i], P5[i], VTIM[i], SPIN[i], [EventWeights for k in range(len(ID[i]))], DER_P_T[i], DER_Eta[i], DER_Azmithul_Angle[i]))
         LocEventID = LocEventID + 1
         pbar.update(n=1)
            
    pbar.leave = None
    pbar.close()
    DataSet = pd.DataFrame(WriteList, columns = ['EventID','PDGID','IST','MOTH1','MOTH2','ICOL1','ICOL2','P1','P2','P3','P4','P5','VTIM','SPIN','WEIGHT','DER_P_T','DER_Eta','DER_Azmithul_Angle'])
    return DataSet

def ROOT2CSV(SelectedDirectory):
    """
    Directory containing ROOT files.

    Parameters
    ----------
    SelectedDirectory : String
        Path to the directory containing the ROOT files.

    Returns
    -------
    None.

    """
    EventID = 1
    SelectedDirectory = os.path.normpath(SelectedDirectory)
    for root, dirs, files in os.walk(SelectedDirectory):
        for names in files:
             if ".root" in os.path.join(root, names):
                 if EventID == 1:
                     DataFrame = ExtractFromROOT(os.path.join(root, names), EventID)
                     EventID = max(DataFrame.EventID) + 1
                 else:
                     DataFrame = DataFrame.append(ExtractFromROOT(os.path.join(root, names), EventID))
                     EventID = max(DataFrame.EventID) + 1
    print('Found {} events'.format(max(DataFrame.EventID)))
    print('Finished converting. Saving file....')
    print(DataFrame.head()) 
    DataFrame.to_csv(SelectedDirectory + r"\PsuedoRapidityDataSet.csv", index = False)                     
             

def RunAllconversions():
    """
    Converts either all LHE or ROOT files to a set of csv files in the selected directory. It first checks for ROOT files, if none are found it then looks for LHE files.

    Returns
    -------
    SelectedDirectory : String
        Returns a string containing the absolute path to the selected directory.

    """
    SelectedDirectory = fd.askdirectory() 
    USEROOT = ROOTFILES(SelectedDirectory)
    if USEROOT:
       print('Converting ROOT files to CSV')
       ROOT2CSV(SelectedDirectory)  
       print("combining all sub events in the events dataset.")
       RecombineEvents(SelectedDirectory)
    else:
        print("Converting txt file to csv format.")
        CreateFile(SelectedDirectory)  
        print("Adding the derived features to the dataset.")
        ConvertoPseudorapidity(SelectedDirectory)        
        print("combining all sub events in the events dataset.")    
        RecombineEvents(SelectedDirectory)
    return SelectedDirectory
        


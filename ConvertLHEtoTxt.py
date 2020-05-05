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
import tkinter as tk
from tkinter import filedialog as fd

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
#print("Files in %r: %s" % (cwd, files))

def ExtractInfo(String):
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
    return np.sqrt(P1**2+P2**2)

def ValueEta(P1, P2, P3):
    CosTheta = P3/np.sqrt(P1**2  + P2**2 + P3**2)
    return -0.5*np.log((1-CosTheta)/(1+CosTheta))

def ValueAzmithulAngle(P1, P2, P3, P_T):
    return np.arcsin(P_T/np.sqrt(P1**2  + P2**2 + P3**2))

def CreateFile(Folder_selected):
    NewFile = open(Folder_selected + r'\LHEEventData.csv','w+', newline='')
    writer = csv.writer(NewFile)
    writer.writerow(["EventID","PDGID","IST","MOTH1","MOTH2","ICOL1","ICOL2","P1","P2","P3","P4","P5","VTIM","SPIN"])
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
    File = open(LHEFile, 'r')
    LineText = File.readlines()
    del LineText[0:LineText.index("</init>\n") +1 ]
    WriteList = []
    LocEventID = EventID
    pbar = tqdm(total = len(LineText), leave = None)
    
    while len(LineText) !=1 :
        try:
            for x in LineText[LineText.index('<event>\n')+2:LineText.index('<mgrwt>\n')]:
                WriteList.append([LocEventID] + ExtractInfo(x))
        except:
            for x in LineText[LineText.index('<event>\n')+2:LineText.index('</event>\n')]:
                WriteList.append([LocEventID] + ExtractInfo(x))
        pbar.update(LineText.index('</event>\n') +1 - LineText.index('<event>\n'))
        del LineText[LineText.index('<event>\n'):LineText.index('</event>\n') +1] 
        LocEventID = LocEventID + 1   
            
    pbar.leave = None
    pbar.update(len(LineText))   
    pbar.close()   
    File.close()
    return [LocEventID, WriteList]


def ConvertoPseudorapidity(SelectedDirectory):
    DataSet = pd.read_csv(SelectedDirectory + r'\LHEEventData.csv' )
    print(DataSet.head())
    Temp = np.zeros((len(DataSet),3))
    for i in tqdm(range(len(DataSet)), leave = None):
        Temp[i,0] = ValueP_T(DataSet.P1[i], DataSet.P2[i])
        Temp[i,1] = ValueEta(DataSet.P1[i],DataSet.P2[i], DataSet.P3[i])
        Temp[i,2] = ValueAzmithulAngle(DataSet.P1[i], DataSet.P2[i], DataSet.P3[i], ValueP_T(DataSet.P1[i], DataSet.P2[i]))
    DataSet.insert(loc = len(DataSet.columns), column = "DER_P_T", value = Temp[:,0])
    DataSet.insert(loc = len(DataSet.columns), column = "DER_Eta", value = Temp[:,1])    
    DataSet.insert(loc = len(DataSet.columns), column = "DER_Azmithul_Angle", value = Temp[:,2])
    DataSet.to_csv(SelectedDirectory + r"\PsuedoRapidityDataSet.csv", index = False)
    print(DataSet.head())   
    
    
def RecombineEvents(SelectedDirectory):
    PDGID_Lepton_List = [11, 12, 13, 14, 15, 16, 17, 18]
    PDGID_Quark_List =  [1, 2, 3, 4, 5, 6, 7, 8]
    PDGID_Boson_List = [9, 21, 22, 23, 24, 25, 37]
    ParticleIDofinterest = [-6, 6]
    DataSet = pd.read_csv(SelectedDirectory + r"\PsuedoRapidityDataSet.csv")
    EventDataSet = {"EventID" : [],
                    "DER_No_Detected_Particles": [],
                    "DER_No_electrons": [],
                    "DER_No_muons": [],
                    "DER_No_Leptons":[],
                    "DER_PT_Tot_Detected_Particles":[],
                    "DER_Missing_Momentum": [],
                    "DER_Momentum_of_detected_Leptons":[],
                    "Label" : []}
        
    for i in tqdm(np.unique(DataSet.EventID), leave = None):   
        Temp = DataSet[DataSet.EventID == i]
        EventDataSet["EventID"].append(i)
        EventDataSet["DER_No_Detected_Particles"].append(len(Temp[Temp.IST ==1 ]))
        EventDataSet["DER_No_electrons"].append(len(Temp[(Temp.IST == 1) & (abs(Temp.PDGID) == 11)]))
        EventDataSet["DER_No_muons"].append(len(Temp[(Temp.IST == 1) & (abs(Temp.PDGID) == 13)]))
        EventDataSet["DER_No_Leptons"].append(len(Temp[(Temp.IST == 1) & (abs(Temp.PDGID).isin(PDGID_Lepton_List) )]))
        EventDataSet["DER_PT_Tot_Detected_Particles"].append(sum(Temp.DER_P_T[(Temp.IST == 1) & (abs(Temp.PDGID) != 12)]))
        EventDataSet["DER_Missing_Momentum"].append(sum(Temp.DER_P_T[(Temp.IST == 1) & (abs(Temp.PDGID) == 12)]))
        EventDataSet["DER_Momentum_of_detected_Leptons"].append(sum(Temp.DER_P_T[(Temp.IST == 1) & (abs(Temp.PDGID).isin([PDGID_Lepton_List]))]))
        if any(item in list(Temp.PDGID[Temp.IST == 1]) for item in ParticleIDofinterest):
            EventDataSet['Label'].append(1)
        else:
             EventDataSet['Label'].append(0)
       
    
    df = pd.DataFrame(EventDataSet)
    print("Sample of the data.")
    print(df.head())
    df.to_csv(SelectedDirectory + "\EventData.csv", index = False)    
    
def RunAllconversions():
    SelectedDirectory = fd.askdirectory() 
    #folder_selected = fd.askdirectory(title = "Location of the run folders", prompt = "Selected the folder containing the run files.")
    i = 0 
    for root, dirs, files in os.walk(SelectedDirectory):
        for name in files:
            if "unweighted_events.lhe" in os.path.join(root, name):
              if "unweighted_events.lhe.gz" not in os.path.join(root, name):   
                 i = i + 1
    assert(i != 0 ), "No data files. Ensure that you have selected a folder containing the files 'unweighted_events.lhe'."
    print("Found {} data unweighted event files.\n".format(i))
    print("Converting txt file to csv format.")
    CreateFile(SelectedDirectory)  
    print("Adding the derived features to the dataset.")
    ConvertoPseudorapidity(SelectedDirectory)        
    print("Combinig combining all sub events in the events dataset.")    
    RecombineEvents(SelectedDirectory)
    return SelectedDirectory
        
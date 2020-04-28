# Physics-Machine-Learning-project
These are the files that I have used to generate the results that are used in the Machine Learning Project at UJ.

This repository contains Python 3 code which has been used to study the files produced from MADGraph 5 particle decay simulations. At this point, the files are designed to run using the datasets produced from unweighted events.

## ConvertToTxt.py
The ConvertToTxt file uses the *"unweighted_events.lhe"* files to extract the information about the events produced and rewrites them in the form of several .csv files. The procedure **RunAllconversions** should be used to extract all the data in its forms. Once the function is run it will prompt the users to select a folder. This folder should contain the "unweighted_events.lhe" files or subfolders that contain the *"unweighted_events.lhe"* files. All produced files will be saved in the selected folder as .csv files. Functions associated with the file.
### RunAllconversions
This function will run all the other procedures and produce all of the files that is it will produce the *"LHEEventData.csv"* file, which is simply the outputted DataSet in .csv form for easier loading of data for machine learning purposes. **Note this is the recommended function to run**. Next, it creates the *"PsuedoRapidityDataSet.csv"* this file contains the original data from the MADGraph dataset plus additional features which are derived from the original features (Currently the transverse momentum **P_T** and the **pseudorapidity angle eta**). Finally, the procedure will create a dataset in which all the events are recombined and only information about the possible detected particles is available, the idea of this file is to store the data in a way that would be more applicable to detector data. The features in this file are as follows:
  * **EventID**: A primary key for storing the data.
  * **DER_No_Detected_Particles**: Number of actually detected particles.
  * **DER_No_unique_Particles**: Number of unique detected particles.   
  * **DER_No_Leptons**: Number of detected Leptons.
  * **DER_No_Bosons**: Number of detected Bosons.
  * **DER_PT_Tot_Detected_Particles**: The summed momentum of the detected particles.
  * **DER_Delta_eta**: The difference in the pseudorapidity angle of the first and second jet.
  * **DER_Momentum_of_detected_Bosons**: Summed momentum of the Bosons.
  * **DER_Momentum_of_detected_Leptons**: Summed momentum of the detected Leptons.
  * **Label**: This label determines if the partiucles of interest are in the current event. It returns 1 if it is a 'Signal' and 0 if it is a 'Background' event.
  
### CreateFile
Uses the function **ConverttoText** to produce the first .csv file which converts the data in *unweighted_events.lhe* to a .csv  file which makes passing the data to a program much easier, as it has been formatted. It will also print out the total number of events found. The input for this function is *Folder_Selected*, which is the folder where the *"unweighted_events.lhe"* are located, even if they are in subfolders. Furthermore, this is where the *"LHEEventData.csv"* file will be saved. The format must be *"C:\\Path\To\Files\Folder_Selected"*.

### ConverttoText
This function extracts the data from the file "unweighted_events.lhe" and converts it to a list which can be stored in the. It uses the function **ExtractInfo** to pull the information and convert it to a list. **Note that users don't need to call this function and should instead only use the CreateFile function**.

### ConvertoPseudorapidity
This function produces a txt file which is stored as a .csv file and uses the data stored in *"LHEEventData"* to derive new quantities. As of the latest version these features are the **Psuedorapdity** and the total transverse momentum of the particles **P_T**. It takes in as its inputs the value **Selected_Folder**, which is the folder in which the file *"LHEEventData.csv"* is found. It is also the location where the file *"PsuedoRapidityDataSet.csv"* will be saved.

### RecombineEvents
This function uses the values in the file *"PsuedoRapidityDataSet.csv"* to recombine the events and produce a dataset of the events with their subprocesses combined. Furthermore, only the particles that could be detected have had their respective features considered. This is the file that should be used for the predictive methods as this file should be the closest to a real dataset as is possible. 

### OtherFunctions
The rest of the functions in this file should not be called directly they should only be called by their respective functions. 

##Feature_Plots_PCA.py
This file contains all the functions for performing feature analysis on the datasets. This includes a function for performing a PCA. It produces plots of the various features, their linear correlation, and a PCA plot to show the weightings of the various features. 

### FeaturePlots
This function produces a pair plot which shows the linear correlation between the features in the dataset. Its inputs are **DataSet** which should be a dataset from the files produced from the file **ConvertLHEToTxt.py**. The DataSet should not contain any infinities in it or any NaN values as this will prevent the feature from being plotted.

### PCAAnalysis
This produces a PCA and plot of the PC's and their variation. The input values are **DataSet**, the dataset without the **LabelOfInterest**, and **LabelOfInterest**, the feature we wish to predict when applying machine learning to the dataset. The DataSet should not contain any infinities in it or any NaN values as this will prevent the feature from being plotted.

### Other Functions
The other functions in this file are called by the above two functions. They should not be called by the user. 

## Shrinkage methods.py
This file contains the possible shrinkage methods that can be used to analyse the data. All the functions produce output accuracy values for the methods and include plots of the weightings of the features in the respective shrinkage methods. The names of the functions describe which statistical method is used as the shrinkage method. The functions all take the **dataset**, without the label of interest, and **Y** a list containing the labels for the data as their arguments.

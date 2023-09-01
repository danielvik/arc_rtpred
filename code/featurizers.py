import pandas as pd 
import numpy as np
#import datamol as dm
import deepchem as dc
import os
import re 
from tqdm import tqdm
import subprocess


######*
######* FUNCTIONS FOR FEATURIZING 
######*

####################################
#* ChemAxon-based LogD calculations
####################################

def LogD_calculations(data, path_to_chemaxon_licence = None):
    ''' 
    ChemAxon based calculation of LogD descriptors. Must be done while having access to the G:\ drive and the C:\nuevolution\KNIME_TEMP folder.
    
    Binning is a way to provide a set of descriptors that more accurately mirror the distribution of probablity around a calculated property, instead of just a single number. It factores in some of the uncertainty in the calculcation.
    
    Parameters
    ---------- 
    data : dataframe 
        SMILES strings for the molecule we wish to calculate the descriptors for
        
    path_to_chemaxon_licence: str
        string with the file path for chemaxon licence to cx_calc.

    Returns
    -------
    logd_desc : dataframe
        dataframe with the SMILES and calculated logd values. If binning matrix is specified there is also binning descriptors 

    '''
    
    ##* these are hardcoded variables for property calculcations
    os.environ['CHEMAXON_LICENSE_URL']= path_to_chemaxon_licence

    exe_file_path = r"C:\nuevolution\knime_temp\ChemAxon\MarvinSuite\bin\cxcalc"
    input_file_path = r"C:\nuevolution\KNIME_TEMP\smiles_for_chemaxon_logd_calc.smiles"
    output_file_path = r"C:\nuevolution\KNIME_TEMP\ChemAxon_LogD_out.csv"
    
    ##* saving data for the commandline tool to access it
    data.to_csv(input_file_path, index=False, header=False)
    
    ##* running the commandline tool
    filepath = f'{exe_file_path} -g logd -m weighted -H 7.4 logd -H 7 logd -H 6.5 logd -H 6 logd -H 5.5 logd -H 5 logd -H 4.5 logd -H 4 logd -H 3.5 logd -H 3 logd -H 2.5 logd -H 2 logd -H 1.5 logd -H 1 logd -H 0.5 logd -H 0 "{input_file_path}" > "{output_file_path}"'
    subprocess.run(filepath, shell=True)
    
    ##* reading the output file
    logd_desc = pd.read_csv(output_file_path, sep = '\t')
    logd_desc = logd_desc.replace(',','.', regex=True).apply(pd.to_numeric)
    logd_desc = pd.concat([data, logd_desc], axis = 1).drop(['id'], axis = 1)
   
    ##* extracting the binning descriptors 
    if isinstance(path_to_binning_matrix,str):
        
        binning_matix = pd.read_csv(path_to_binning_matrix)

        logd_bins = pd.DataFrame()

        for i,smile in enumerate(logd_desc.SMILES):
            logd_45 = logd_desc['logD[pH=4.5]'][i]
            row_w_bins = binning_matix[binning_matix['LogD4_5'] == round(logd_45,1)]
            logd_bins = pd.concat([logd_bins,row_w_bins], axis = 0, ignore_index=True)
            
        logd_desc = pd.concat([logd_desc, logd_bins], axis = 1)
    
    ##* output the descriptors
    return logd_desc
   


####################################
#* DeepChem-based ECFP4 fingerprints
####################################

#? function to make a deepchem diskdataset with option for count-based (must then incl normalization)

#? function to convert diskdataset to .CSV


####################################
#* DeepChem-based RDKit Descriptors
####################################

#? function to make a deepchem diskdataset

#? function to convert diskdataset to .CSV

####################################
#* DeepChem-based MolGraphConv
####################################

#? function to make a deepchem diskdataset

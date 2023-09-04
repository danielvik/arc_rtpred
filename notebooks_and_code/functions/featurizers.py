import pandas as pd 
import numpy as np
#import datamol as dm
import deepchem as dc
import os
import re 
from tqdm import tqdm
import subprocess
import csv
from rdkit import Chem
#from functions.featurizers import LogD_calculations



######*
######* FUNCTIONS FOR FEATURIZING 
######*

####################################
#* Get METLIN SMRT data from remote URL
####################################

def get_METLIN_data(sampling = 2000, only_retained = True):
    ''' Function to pull METLIN SMRT data from remote URL and save subset as a csv file'''
    #* pulling data from a URL
    url = 'https://figshare.com/ndownloader/files/18130628'

    #* read the data into a pandas dataframe and convert inchi to SMILES
    df = pd.read_csv(url, sep=';').rename(columns = {'pubchem':'id'})
    df['mol'] = [Chem.MolFromInchi(x) for x in df['inchi']]
    df['smiles'] = [Chem.MolToSmiles(mol) if mol is not None else None for mol in df['mol']]
    df = df.dropna(axis = 0) #* dropping rows with NaN values as some mols did not convert to SMILES

    #* removing non-retained compounds
    if only_retained:
        df = df[df['rt'] > 200]

    #* sampling the data
    df_subset = df[['id','smiles','rt']].sample(sampling)

    #* outputting the data in a csv file
    output_dir = '../data/metlin_smrt'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'sample_dataset_{sampling}_datapoints.csv')    
    df_subset.to_csv(output_path, index=False)
    print('saved to: ', output_path,'\n')
    print(df_subset.head(3))

    return output_path

####################################
#* ChemAxon-based LogD calculations
####################################

def LogD_calculations(data, path_to_chemaxon_licence = None, path_to_chemaxon_tools = None):
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

    exe_file_path = os.path.join(path_to_chemaxon_tools, "ChemAxon/MarvinSuite/bin/cxcalc")
    input_file_path = os.path.join(path_to_chemaxon_tools, 'smiles_for_chemaxon_logd_calc.smiles')
    output_file_path = os.path.join(path_to_chemaxon_tools, 'ChemAxon_LogD_out.csv')

    ##* saving data for the commandline tool to access it
    data.to_csv(input_file_path, index=False, header=False)
    
    ##* running the commandline tool
    filepath = f'{exe_file_path} -g logd -m weighted -H 7.4 logd -H 7 logd -H 6.5 logd -H 6 logd -H 5.5 logd -H 5 logd -H 4.5 logd -H 4 logd -H 3.5 logd -H 3 logd -H 2.5 logd -H 2 logd -H 1.5 logd -H 1 logd -H 0.5 logd -H 0 "{input_file_path}" > "{output_file_path}"'
    subprocess.run(filepath, shell=True)
    
    ##* reading the output file
    logd_desc = pd.read_csv(output_file_path, sep = '\t')
    # drop rows with 'logd:FAILED' values 
    logd_desc = logd_desc[~logd_desc['logD[pH=7.4]'].str.contains('FAILED')]
    # drop rows with NA values
    logd_desc = logd_desc.dropna(axis = 0)
                            
    logd_desc = logd_desc.replace(',','.', regex=True).apply(pd.to_numeric)
    logd_desc = pd.concat([data, logd_desc], axis = 1).drop(['id'], axis = 1)
   
    
    ##* output the descriptors
    return logd_desc
   

####################################
#* Get all features 
####################################

def get_features(path_to_data, feature_dir, feature_list, path_to_chemaxon_licence = None, path_to_chemaxon_tools = None):
    '''function to get the features from the input feature_list. Features are saved to a csv file in the feature_dir'''
    
    # ECFP4 arguments
    nBits = 2048
    radius = 2
    
    #feature_list = ['logD', 'ecfp4_csv', 'ecfp4_disk', 'rdkit_csv', 'rdkit_disk', 'molgraphconv']

    
    data = pd.read_csv(path_to_data)

    
    ###* LOG D Calculations
    if 'logD' in feature_list and path_to_chemaxon_licence is not None:
        target_dir = os.path.join(feature_dir,'logd_calculations')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        ##* the SMILES must be saved in a flat file for the cmdline tool to read them 
        # subsetting the test data, to save time
        data_for_calc = data[['smiles']].reset_index(drop=True)

        print('ChemAxon-based LogD calculations - to .CSV')

        desc = LogD_calculations(data_for_calc, path_to_chemaxon_licence=path_to_chemaxon_licence, path_to_chemaxon_tools = path_to_chemaxon_tools)

        desc = data.merge(desc, on = 'smiles', how = 'left')

        desc.to_csv(os.path.join(target_dir, 'all_data.csv'), index=False)


        NA_alert = desc.isna().any().any()
        if NA_alert:
            print('*** NA values in:',target_dir)

            
            
    ###* ECFP4 Calculations
    if 'ecfp4' in feature_list: 
    
        target_dir = os.path.join(feature_dir,'ecfp4_disk')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        print('ECFP4 Featurization - to DiskDataset')

        ## creating a deepchem dataset with featurized smiles
        featurizer = dc.feat.CircularFingerprint(size=nBits, radius=radius)
        feats = featurizer.featurize(data.smiles)
        dataset = dc.data.DiskDataset.from_numpy(feats, 
                                                data.rt, 
                                                tasks = ['RT_sec'], 
                                                ids = data.smiles,
                                                data_dir = os.path.join(target_dir,'all_data'))
    
    
        target_dir = os.path.join(feature_dir,'ecfp4_csv')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print('ECFP4 Featurization - to .CSV')
        feats_df = dataset.to_dataframe()
        feats_df.to_csv(os.path.join(target_dir, 'all_data.csv'),index = False)

        NA_alert = feats_df.isna().any().any()
        if NA_alert:
            print('*** NA values in:',target_dir)



    ###* RDKit Calculations
    if 'rdkit' in feature_list:
        target_dir = os.path.join(feature_dir,'rdkit_disk')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            

        print('RDKit Descriptors - to diskdataset')
        ## creating a deepchem dataset with featurized smiles
        featurizer = dc.feat.RDKitDescriptors(is_normalized = True, use_bcut2d = False)
        feats = featurizer.featurize(data.smiles)
        dataset = dc.data.DiskDataset.from_numpy(feats, 
                                                data.rt, 
                                                tasks = ['RT_sec'], 
                                                ids = data.smiles,
                                                data_dir = os.path.join(target_dir,'all_data'))

        target_dir = os.path.join(feature_dir,'rdkit_csv')
        if not os.path.exists(target_dir):
            
            os.makedirs(target_dir)

        print('RDKit Descriptors - to .CSV')
        feats_df = dataset.to_dataframe()
        feats_df = feats_df.fillna(0)

        feats_df.to_csv(os.path.join(target_dir, 'all_data.csv'),index = False)

        NA_alert = feats_df.isna().any().any()
        if NA_alert:
            print('*** NA values in:',target_dir)


    ###* MolGraphConv 
    if 'molgraphconv' in feature_list:
        target_dir = os.path.join(feature_dir,'molgraphconv')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print('MolGraphConv Feat - to diskdataset')
        ## creating a deepchem dataset with featurized smiles
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        feats = featurizer.featurize(data.smiles)
        dataset = dc.data.DiskDataset.from_numpy(feats, 
                                                data.rt, 
                                                tasks = ['RT_sec'], 
                                                ids = data.smiles,
                                                data_dir = os.path.join(target_dir,'all_data'))
    
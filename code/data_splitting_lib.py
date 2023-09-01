import pandas as pd 
import numpy as np
#import datamol as dm
import deepchem as dc
import os
import re 
from tqdm import tqdm



def feature_splitter_csv(path_to_features,path_to_splits, save_folder_name = 'data_splits'):
    
    ##*  collecting the needed paths for the features

    csv_feature_paths = []
    csv_feature_subdirs = []

    for subdir, dirs, files in os.walk(path_to_features):
        for file in files:
            # csv-based features
            if (file.startswith('all_data') and file.endswith('.csv')):
                csv_feature_paths.append(os.path.join(subdir, file))
                csv_feature_subdirs.append(subdir)
            # deepchem diskdatasets
            #elif (file.endswith('.gzip')):
            #    disk_feature_subdirs.append(subdir)

    ##* getting the paths for the premade-splits

    split_paths = []
    output_paths = []

    for subdir, dirs, files in os.walk(path_to_splits):
        for file in files:
            if (file.endswith('.csv')):
                split_paths.append(os.path.join(subdir, file))
                output_paths.append(file)
    
    
    ##* walking through each .CSV feature dir

    for i, path in enumerate(csv_feature_paths):
        features = pd.read_csv(path).rename(columns={'y':'RT', 'adj_RT_sec':'RT','ids':'NUEVO_ID'})
        
        #two types of feature output: Deepchem generated
        if 'w' in features:
            print('Deepchem features:',path)
            features = features.drop(['w', 'RT'], axis = 1)

        # or Chemaxon (cxcalc) generated    
        elif 'logD[pH=4.5]' in features:
            print('cxcalc features:',path)
            features = features.drop(['SMILES','INCHI_KEY','RT'], axis = 1)

        ##* for each feature type, the features are split according to the premade_splits
        for j, split in tqdm(enumerate(split_paths)):
            split = pd.read_csv(split_paths[j])
            merged_split = split.merge(features, on = 'NUEVO_ID', how = 'inner')
 
            save_dir = os.path.join(csv_feature_subdirs[i], save_folder_name) #!
            if not os.path.exists(save_dir):#!
                os.makedirs(save_dir) #!
            print(f'*** Saving data splits to {save_dir} ***')    
            
            labels_split = merged_split[['SMILES','RT']]
#            labels_split.to_csv(os.path.join(csv_feature_subdirs[i], ('labels_'+output_paths[j])), index=False)
            labels_split.to_csv(os.path.join(save_dir, ('labels_'+output_paths[j])), index=False)
            
            feature_split = merged_split.drop(['NUEVO_ID','RT','SMILES'], axis = 1)
            if 'w' in feature_split: 
                feature_split = feature_split.drop(['w'], axis =1)
#            feature_split.to_csv(os.path.join(csv_feature_subdirs[i], ('features_'+output_paths[j])), index=False)
            feature_split.to_csv(os.path.join(save_dir, ('features_'+output_paths[j])), index=False)




def feature_splitter_diskdatasets(path_to_features,path_to_splits, save_folder_name = 'data_splits'):
        
    ##*  collecting the needed paths for the features

    disk_feature_subdirs = []

    for subdir, dirs, files in os.walk(path_to_features):
        for file in files:
            # csv-based features
            if (file.endswith('.gzip') and ('all_data' in subdir)):
                disk_feature_subdirs.append(subdir)

    ##* getting the paths for the premade-splits

    split_paths = []
    output_paths = []

    for subdir, dirs, files in os.walk(path_to_splits):
        for file in files:
            if (file.endswith('.csv')):
                split_paths.append(os.path.join(subdir, file))
                output_paths.append(file)
    
    for i, path in enumerate(disk_feature_subdirs):
        print(i, path)

        ##* loading all datapoints for the specific feature set    
        dataset = dc.data.DiskDataset(data_dir=path)
        array = dataset.ids

        ##* for each feature type, the features are split according to the premade_splits
        for j, split in tqdm(enumerate(split_paths)):
            
            # loading split dataset and extracts IDs
            split = pd.read_csv(split_paths[j])               
            df_ids = split['NUEVO_ID'].values

            # matches with the array to find indexes of datapoints present in current split
            matches = np.where(np.isin(array, df_ids))[0]

            # defining a folder name based on split name
            split_name = re.search(r'(.*).csv$', output_paths[j])[1]
            feat_dir = re.search(r'features\\(.*)\\all_data', path)[1]
            output_dir = os.path.join(path_to_features,feat_dir,save_folder_name, split_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f'*** Saving data splits to {output_dir} ***')    
            # using the deepchem .select() function to subset the dataset according to matches
            dataset.select(matches, select_dir=output_dir)

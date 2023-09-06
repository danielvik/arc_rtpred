import pandas as pd
import numpy as np
import deepchem as dc
import os
import re


def data_splitter(path_to_data, destination_dir, number_of_splits=5, seed=42):
    ##* Loading data into diskDataset and using scaffold-split
    data = pd.read_csv(path_to_data)

    dataset = dc.data.NumpyDataset(X=data.id, y=data.rt, ids=data.smiles)

    splitter = dc.splits.ScaffoldSplitter()

    train_dataset, test_dataset = splitter.train_test_split(
        dataset=dataset, frac_train=0.9, seed=seed
    )

    ##* saving test and train splits as .CSV
    train_df = train_dataset.to_dataframe()
    train_df = train_df.rename(columns={"X": "id", "y": "rt", "ids": "smiles"})
    train_df.to_csv(os.path.join(destination_dir, "train_df.csv"), index=False)

    test_df = test_dataset.to_dataframe()
    test_df = test_df.rename(columns={"X": "id", "y": "rt", "ids": "smiles"})
    test_df.to_csv(os.path.join(destination_dir, "test_df.csv"), index=False)

    ##* Preparing Cross validation splits
    target_dir = os.path.join(destination_dir, "cv_splits")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    cv_splits = splitter.k_fold_split(dataset=train_dataset, k=number_of_splits)

    for i, k_split in enumerate(cv_splits):
        train_k_split = k_split[0]
        train_k_split = train_k_split.to_dataframe()
        train_k_split = train_k_split.rename(
            columns={"X": "id", "y": "rt", "ids": "smiles"}
        )
        train_k_split.to_csv(
            os.path.join(target_dir, f"train_{i}_split.csv"), index=False
        )

        valid_k_split = k_split[1]
        valid_k_split = valid_k_split.to_dataframe()
        valid_k_split = valid_k_split.rename(
            columns={"X": "id", "y": "rt", "ids": "smiles"}
        )
        valid_k_split.to_csv(
            os.path.join(target_dir, f"valid_{i}_split.csv"), index=False
        )


def feature_splitter_csv(
    path_to_features, path_to_splits, save_folder_name="data_splits"
):
    ##*  collecting the needed paths for the features

    csv_feature_paths = []
    csv_feature_subdirs = []

    for subdir, dirs, files in os.walk(path_to_features):
        for file in files:
            # csv-based features
            if file.startswith("all_data") and file.endswith(".csv"):
                csv_feature_paths.append(os.path.join(subdir, file))
                csv_feature_subdirs.append(subdir)

    ##* getting the paths for the premade-splits

    split_paths = []
    output_paths = []

    for subdir, dirs, files in os.walk(path_to_splits):
        for file in files:
            if file.endswith(".csv"):
                split_paths.append(os.path.join(subdir, file))
                output_paths.append(file)

    ##* walking through each .CSV feature dir

    for i, path in enumerate(csv_feature_paths):
        features = pd.read_csv(path).rename(columns={"y": "rt", "ids": "id"})

        # two types of feature output: Deepchem generated
        if "w" in features:
            features = features.drop(["w", "rt"], axis=1)

        # or Chemaxon (cxcalc) generated
        elif "logD[pH=4.5]" in features:
            features = features.drop(["smiles", "rt"], axis=1)

        ##* for each feature type, the features are split according to the premade_splits
        for j, split in enumerate(split_paths):
            split = pd.read_csv(split_paths[j])
            merged_split = pd.concat([split, features], axis=1, join="inner")

            save_dir = os.path.join(csv_feature_subdirs[i], save_folder_name)  #!
            if not os.path.exists(save_dir):  #!
                os.makedirs(save_dir)  #!

            labels_split = merged_split[["smiles", "rt"]]
            labels_split.to_csv(
                os.path.join(save_dir, ("labels_" + output_paths[j])), index=False
            )

            feature_split = merged_split.drop(["id", "rt", "smiles"], axis=1)
            if "w" in feature_split:
                feature_split = feature_split.drop(["w"], axis=1)
            feature_split.to_csv(
                os.path.join(save_dir, ("features_" + output_paths[j])), index=False
            )


def feature_splitter_diskdatasets(
    path_to_features, path_to_splits, save_folder_name="data_splits"
):
    ##*  collecting the needed paths for the features

    disk_feature_subdirs = []

    for subdir, dirs, files in os.walk(path_to_features):
        for file in files:
            # csv-based features
            if file.endswith(".gzip") and ("all_data" in subdir):
                disk_feature_subdirs.append(subdir)

    ##* getting the paths for the premade-splits

    split_paths = []
    output_paths = []

    for subdir, dirs, files in os.walk(path_to_splits):
        for file in files:
            if file.endswith(".csv"):
                split_paths.append(os.path.join(subdir, file))
                output_paths.append(file)

    for i, path in enumerate(disk_feature_subdirs):
        path = os.path.abspath(path)
        ##* loading all datapoints for the specific feature set
        dataset = dc.data.DiskDataset(data_dir=path)
        array = dataset.ids

        ##* for each feature type, the features are split according to the premade_splits
        for j, split in enumerate(split_paths):
            # loading split dataset and extracts IDs
            split = pd.read_csv(split_paths[j])
            df_ids = split["smiles"].values

            # matches with the array to find indexes of datapoints present in current split
            matches = np.where(np.isin(array, df_ids))[0]

            # defining a folder name based on split name
            split_name = re.search(r"(.*).csv$", output_paths[j])[1]
            feat_dir = re.search(r"features\\(.*)\\all_data", path)[1]
            output_dir = os.path.join(
                path_to_features, feat_dir, save_folder_name, split_name
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # using the deepchem .select() function to subset the dataset according to matches
            dataset.select(matches, select_dir=output_dir)

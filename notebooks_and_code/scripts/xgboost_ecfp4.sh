#!/bin/bash

conda activate deepchem_env

export model_dir=/models/xgboost_ecfp4/
mkdir -p "$model_dir"

export base_data_dir=./data/metlin_smrt/features/ecfp4_csv/cv_splits/

# looping through the 5 CV splits
start=0 \
end=4

# Iterate using a for loop
for ((i=start; i<=end; i++))
do

    export model_dir=/models/xgboost_ecfp4/xgboost_ecfp4_${i}/
    mkdir ${model_dir}


    python ./notebooks_and_code/functions/model_training/xgboost_hyperopt.py \
    --train_labels ${base_data_dir}/labels_train_${i}_split.csv/ \
    --train_feats ${base_data_dir}/features_train_${i}_split.csv/ \
    --valid_labels ${base_data_dir}/labels_valid_${i}_split.csv/ \
    --valid_feats ${base_data_dir}/features_valid_${i}_split.csv/ \
    --test_labels ${base_data_dir}/labels_test_df.csv/ \
    --test_feats ${base_data_dir}/features_test_df.csv/ \
    --model_directory ${model_dir} \
    --epochs 100 \
    --iterations 20

done



#!/bin/bash



export model_dir=/models/chemprop_ecfp4/
mkdir -p "$model_dir"

export base_data_dir=./data/metlin_smrt/features/ecfp4_csv/cv_splits/

start=0 \
end=4

# Iterate using a for loop
for ((i=start; i<=end; i++))
do

    export model_dir=/models/chemprop_ecfp4/chemprop_ecfp4_hyper_${i}/
    mkdir ${model_dir}

    ###

    chemprop_hyperopt \
    --data_path ${base_data_dir}/labels_train_${i}_split.csv \
    --features_path ${base_data_dir}/features_train_${i}_split.csv \
    --separate_val_path ${base_data_dir}/labels_valid_${i}_split.csv \
    --separate_val_features_path ${base_data_dir}/features_valid_${i}_split.csv \
    --no_features_scaling \
    --dataset_type regression \
    --log_dir ${model_dir}\
    --config_save_path ${model_dir}/config.json \
    --hyperopt_checkpoint_dir ${model_dir} \
    --metric mse \
    --extra_metrics mae rmse r2 \
    --save_preds \
    --epochs 100 \
    --num_iters 20 \
    --no_cache_mol \
    --num_workers 20

    ###

    chemprop_train \
    --data_path ${base_data_dir}/labels_train_${i}_split.csv \
    --features_path ${base_data_dir}/features_train_${i}_split.csv \
    --separate_val_path ${base_data_dir}/labels_valid_${i}_split.csv \
    --separate_val_features_path ${base_data_dir}/features_valid_${i}_split.csv \
    --separate_test_path ${base_data_dir}/features_test_df.csv \
    --separate_test_features_path ${base_data_dir}/labels_test_df.csv \
    --no_features_scaling \
    --dataset_type regression \
    --config_path ${model_dir}/config.json\
    --save_dir ${model_dir} \
    --metric mse \
    --extra_metrics mae rmse r2 \
    --save_preds \
    --epochs 100 \
    --no_cache_mol \
    --num_workers 20

done

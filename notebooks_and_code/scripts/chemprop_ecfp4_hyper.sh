#!/bin/bash


source /compchem/arc/apps/anaconda/anaconda-2020.11_dkcn-plws-arc02/etc/profile.d/conda.sh

conda activate chemprop_env

# Define the range of iterations
start=0 \
end=4

export temporaryDirectory=/scratch/arc/${SLURM_JOB_ID}/
mkdir ${temporaryDirectory}

# Iterate using a for loop
for ((i=start; i<=end; i++))
do

    export temporaryDirectory=/scratch/arc/${SLURM_JOB_ID}/chemprop_ecfp4_hyper_${i}/
    mkdir ${temporaryDirectory}

    export base_data_dir=/compchem/arc/users/dvik/rt_pub/data/features/ecfp4_csv/cv_splits/


    ###

    chemprop_hyperopt \
    --data_path ${base_data_dir}/labels_train_${i}_split.csv \
    --features_path ${base_data_dir}/features_train_${i}_split.csv \
    --separate_val_path ${base_data_dir}/labels_valid_${i}_split.csv \
    --separate_val_features_path ${base_data_dir}/features_valid_${i}_split.csv \
    --no_features_scaling \
    --dataset_type regression \
    --log_dir ${temporaryDirectory}\
    --config_save_path ${temporaryDirectory}/config.json \
    --hyperopt_checkpoint_dir ${temporaryDirectory} \
    --metric mse \
    --extra_metrics mae rmse r2 \
    --save_preds \
    --epochs 100 \
    --num_iters 20 \
    --no_cache_mol \
    --num_workers 20

    cp -r ${temporaryDirectory} /compchem/arc/users/dvik/rt_pub/models/chemprop_ecfp4/

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
    --config_path ${temporaryDirectory}/config.json\
    --save_dir ${temporaryDirectory} \
    --metric mse \
    --extra_metrics mae rmse r2 \
    --save_preds \
    --epochs 100 \
    --no_cache_mol \
    --num_workers 20

    cp -r ${temporaryDirectory} /compchem/arc/users/dvik/rt_pub/models/chemprop_ecfp4/

done

###? to run in command line
###? sbatch -n 5 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/chemprop_ecfp4_hyper.sh  

#! ID: 14811

###? see the output 
###? tail -f slurm-14811.out
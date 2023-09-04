#!/bin/bash


export model_dir=/models/attentivefp/
mkdir -p "$model_dir"
    
export base_data_dir=./data/metlin_smrt/features/molgraphconv/cv_splits/

# Define the range of iterations
start=0 \
end=4

# Iterate using a for loop
for ((i=start; i<=end; i++))
do

    export model_dir=/models/attentivefp/attentivefp_${i}/
    mkdir ${model_dir}
    
    python ./notebooks_and_code/functions/model_training/attentivefp_hyperopt.py \
    --train_dir ${base_data_dir}/train_${i}_split/ \
    --val_dir ${base_data_dir}/valid_${i}_split/ \
    --test_dir ${base_data_dir}/test_df/ \
    --model_directory ${model_dir} \
    --epochs 100 \
    --callback_intervals 1000 \
    --iterations 20

done
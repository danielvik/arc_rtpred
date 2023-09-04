#!/bin/bash

source /compchem/arc/apps/anaconda/anaconda-2020.11_dkcn-plws-arc02/etc/profile.d/conda.sh

conda activate deepchem_env

# SLURM specific stuff:
#  template for submitting jobs to the SLURM queueing system
#  to execute the script:
#  "sbatch -n <numberCpus> -p normal --nodelist=dkcn-papp-nudk1 template_squeue.slm"
#
# SLURM_JOB_ID and SLURM_NNODES are variables from SLURM that can be used
# Define the range of iterations
start=0 \
end=4

export temporaryDirectory=/scratch/arc/${SLURM_JOB_ID}/
mkdir ${temporaryDirectory}

# Iterate using a for loop
for ((i=start; i<=end; i++))
do

    export temporaryDirectory=/scratch/arc/${SLURM_JOB_ID}/xgboost_logd_${i}/
    mkdir ${temporaryDirectory}

    #SLURM_NNODES=${numberCpus}
    export base_data_dir=/compchem/arc/users/dvik/rt_pub/data/features/logd_calculations/cv_splits/



    #mkdir ${temporaryDirectory}

    #conda activate deepchem_env

    python /compchem/arc/users/dvik/rt_pub/code/xgboost_csv.py \
    --train_labels ${base_data_dir}/labels_train_${i}_split.csv/ \
    --train_feats ${base_data_dir}/features_train_${i}_split.csv/ \
    --valid_labels ${base_data_dir}/labels_valid_${i}_split.csv/ \
    --valid_feats ${base_data_dir}/features_valid_${i}_split.csv/ \
    --test_labels ${base_data_dir}/labels_test_df.csv/ \
    --test_feats ${base_data_dir}/features_test_df.csv/ \
    --model_directory ${temporaryDirectory} \
    --epochs 100 \
    --iterations 20


    cp -r ${temporaryDirectory} /compchem/arc/users/dvik/rt_pub/models/xgboost_logd/
done
###? to run in command line
###? sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/xgboost_logd.sh  

#! ID: 15162

###? see the output 
###? tail -f slurm-15162.out


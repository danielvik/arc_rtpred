#!/bin/bash

source /compchem/arc/apps/anaconda/anaconda-2020.11_dkcn-plws-arc02/etc/profile.d/conda.sh

conda activate deepchem_env

# Define the range of iterations
start=0 \
end=4

export temporaryDirectory=/scratch/arc/${SLURM_JOB_ID}/
mkdir ${temporaryDirectory}
    

# Iterate using a for loop
for ((i=start; i<=end; i++))
do
    # SLURM specific stuff:
    #  template for submitting jobs to the SLURM queueing system
    #  to execute the script:
    #  "sbatch -n <numberCpus> -p normal --nodelist=dkcn-papp-nudk1 template_squeue.slm"
    #
    # SLURM_JOB_ID and SLURM_NNODES are variables from SLURM that can be used
    #SLURM_NNODES=${numberCpus}
    export temporaryDirectory=/scratch/arc/${SLURM_JOB_ID}/attentivefp_hyper_${i}/
    mkdir ${temporaryDirectory}
    

    export base_data_dir=/compchem/arc/users/dvik/rt_pub/data/features/molgraphconv/cv_splits/


    ###

    #mkdir ${temporaryDirectory}

    #conda activate deepchem_env

    python /compchem/arc/users/dvik/rt_pub/code/attentivefp_hyperopt.py \
    --train_dir ${base_data_dir}/train_${i}_split/ \
    --val_dir ${base_data_dir}/valid_${i}_split/ \
    --test_dir ${base_data_dir}/test_df/ \
    --model_directory ${temporaryDirectory} \
    --epochs 100 \
    --callback_intervals 1000 \
    --iterations 20


    cp -r ${temporaryDirectory} /compchem/arc/users/dvik/rt_pub/models/attentivefp/

done

###? to run in command line
###? sbatch -n 5 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/attentivefp_hyper.sh  

#! ID: 14998

###? see the output 
###? tail -f slurm-14998.out


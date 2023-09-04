#!/bin/bash

source /compchem/arc/apps/anaconda/anaconda-2020.11_dkcn-plws-arc02/etc/profile.d/conda.sh

conda activate deepchem_env

mkdir /models/

sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/attentivefp.sh 
sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/metlin_model.sh  
sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/chemprop_rdkit_hyper.sh  
sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/chemprop_ecfp4_hyper.sh  
sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/chemprop_logd_hyper.sh  
sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/xgboost_ecfp4.sh  
sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/xgboost_logd.sh  
sbatch -n 20 -p normal --nodelist=dkcn-papp-nudk1 /compchem/arc/users/dvik/rt_pub/code/scripts/xgboost_rdkit.sh  

#!/bin/bash

conda activate arc_rtpred

mkdir /models/

bash  ./notebooks_and_code/scripts/xgboost_ecfp4.sh  
bash  ./notebooks_and_code/scripts/xgboost_logd.sh  
bash  ./notebooks_and_code/scripts/xgboost_rdkit.sh  

bash  ./notebooks_and_code/scripts/attentivefp.sh 
bash  ./notebooks_and_code/scripts/metlin_model.sh  

bash  ./notebooks_and_code/scripts/chemprop_rdkit_hyper.sh  
bash  ./notebooks_and_code/scripts/chemprop_ecfp4_hyper.sh  
bash  ./notebooks_and_code/scripts/chemprop_logd_hyper.sh  

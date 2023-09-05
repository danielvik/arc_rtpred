# arc_rtpred
Code associated with RT prediction manuscript

### Introduction


### Method
We featurize data based on the SMILES strings, and split the data (using scaffold split) first into test(0.1)/train(0.9) and then, additionally, the train set is split in train(0.8)/valid(0.2) for 5-fold cross validation.

The features used are:

- ECFP4, 2048 bit fingerprints
- RDKit descriptors (excl. BCUT2D)
- LogD calculcations for the range pH 0.5-7.4
- Molecular Graph Convolutions

Features are calculcated using DeepChem module (https://deepchem.io/), except for LogD which was done with Chemaxons cxcalc commandline tool (https://docs.chemaxon.com/display/docs/cxcalc-command-line-tool.md).

Data splits and features are saved locally, and used the train a set of models:

- XGBoost
- AttentiveFP
- Fully-connected Neural Network (FCNN)
- ChemProp

Each model training is composed of 5 hyperparameter optimization (100 epochs, 20 iterations) using hyperopt module and TPE search algorithm. The hyperoptimization is then followed by a re-training of the best model settings.

### 


### Requirements
- rdkit
- pytorch
- xgboost
- deepchem
- chemprop
- hyperopt
- dgllife

### Getting started

clone the repo 

create enviroment (env.yml) -- it can take a while to create the env

The  [notebook](./notebooks_and_code/featurizing_and_splitting.ipynb) contains a step-by-step walkthrough of data featurization, splitting, model training and evaluation using the public METLIN SMRT dataset.

After running the notebook, run the [model_queue.sh](./notebooks_and_code/scripts/model_queue.sh) shell script to train the models.

### TODO 

- [ ] make enviroment: conda create env arc_rtpred > yml file or req file
- [x] clean up bash scripts -- 
  - [ ] how to make it generic?
- [x] clean up python scripts
  - [x] attentivefp
  - [x] fcnn
  - [x] xgboost 

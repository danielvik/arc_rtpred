# arc_rtpred

This repo contains the code needed to reproduce the workflow (i.e. data featurization, splitting and model training) reported in the manuscript


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

### Requirements

- numpy v1.20
- scipy v1.9.3
- rdkit v2022.03.5
- pytorch v1.12.1
- xgboost v1.7.6
- deepchem v2.7.1
- chemprop v1.6.0
- hyperopt v0.2.7
- dgllife v0.3.2
- dgl v1.1.2

### Getting started

1. clone the repo: ```git clone https://github.com/danielvik/arc_rtpred.git```
2. ```cd arc_rtpred```
3. create conda environment: ```conda env create -f environment.yml```
4. ```conda activate arc_rtpred```

The build from the yml file is not GPU-enabled, so installing pytorch-gpu and xgboost-gpu should be done after building the environment.


The [notebook](./notebooks_and_code/featurizing_and_splitting.ipynb) contains a step-by-step walkthrough of data featurization, splitting, model training and evaluation using the public METLIN SMRT dataset.

Model training [scripts](./notebooks_and_code/model_training/) can then be run afterwards.
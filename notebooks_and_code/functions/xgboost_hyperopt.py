import pandas as pd
import argparse
import os

# from functions.xgboost_operations import xgboost_hyperopt, xgboost_retrain_test
from xgboost_operations import xgboost_hyperopt, xgboost_retrain_test


seed = 42


############################################################################
# Parameters
############################################################################

parser = argparse.ArgumentParser()

# input/output
parser.add_argument(
    "--train_labels",
    type=str,
    default="../data/metlin_smrt/features/rdkit_csv/cv_splits/labels_train_0_split.csv",
    help="directory for labels to featurized train data -- defaults to split 0",
)
parser.add_argument(
    "--train_feats",
    type=str,
    default="../data/metlin_smrt/features/rdkit_csv/cv_splits/features_train_0_split.csv",
    help="directory for featurized train data -- defaults to split 0",
)

parser.add_argument(
    "--valid_labels",
    type=str,
    default="../data/metlin_smrt/features/rdkit_csv/cv_splits/labels_valid_0_split.csv",
    help="directory for labels to featurized valid data -- defaults to split 0",
)
parser.add_argument(
    "--valid_feats",
    type=str,
    default="../data/metlin_smrt/features/rdkit_csv/cv_splits/features_valid_0_split.csv",
    help="directory for featurized valid data -- defaults to split 0",
)

parser.add_argument(
    "--test_labels",
    type=str,
    default="../data/data/metlin_smrt/features/rdkit_csv/cv_splits/labels_test_df.csv",
    help="directory for labels featurized test data -- defaults to split 0",
)
parser.add_argument(
    "--test_feats",
    type=str,
    default="../data/data/metlin_smrt/features/rdkit_csv/cv_splits/features_test_df.csv",
    help="directory for featurized test data -- defaults to split 0",
)

parser.add_argument(
    "--model_directory",
    type=str,
    default="../models/xgboost/",
    help="directory for model_saving",
)


# training
parser.add_argument(
    "--iterations", type=int, default=20, help="number of hyperopt iterations"
)
parser.add_argument(
    "--epochs", type=int, default=100, help="number of epochs pr iteration"
)

args = parser.parse_args()


print(
    "** This is the XGBoost Regressor, for retention-time prediction, RMSE optimized **"
)

############################################################################
# Reading data and converting data formats
############################################################################

y_train = pd.read_csv(os.path.abspath(args.train_labels))["rt"]
X_train = pd.read_csv(os.path.abspath(args.train_feats))
new_column_names = ["X" + str(i) for i in range(len(X_train.columns))]
X_train.columns = new_column_names


y_valid = pd.read_csv(os.path.abspath(args.valid_labels))["rt"]
X_valid = pd.read_csv(os.path.abspath(args.valid_feats))
new_column_names = ["X" + str(i) for i in range(len(X_valid.columns))]
X_valid.columns = new_column_names


y_test = pd.read_csv(os.path.abspath(args.test_labels))["rt"]
X_test = pd.read_csv(os.path.abspath(args.test_feats))
new_column_names = ["X" + str(i) for i in range(len(X_test.columns))]
X_test.columns = new_column_names


print("*** reloaded the featurized data ***")


############################################################################
# Model training
############################################################################

model_check_point = args.model_directory
epochs = args.epochs
iterations = args.iterations

print("HyperParameter optimization\n")

best_params = xgboost_hyperopt(
    model_check_point,
    X_train,
    y_train,
    X_valid,
    y_valid,
    epochs=epochs,
    iterations=iterations,
)


############################################################################
# Evaluating on test data
############################################################################

print("Model retraining\n")


xgboost_retrain_test(
    best_params, X_train, y_train, X_valid, y_valid, X_test, y_test, model_check_point
)

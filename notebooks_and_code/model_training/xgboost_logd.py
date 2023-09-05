import os
import subprocess

model_dir = "./models/xgboost_logd/"
os.makedirs(model_dir, exist_ok=True)

base_data_dir = "./data/metlin_smrt/features/logd_calculations/cv_splits/"

# Loop through the 5 CV splits
start = 0
end = 4

for i in range(start, end + 1):
    model_subdir = os.path.join(model_dir, f"xgboost_logd_{i}/")
    os.makedirs(model_subdir, exist_ok=True)

    subprocess.run(
        [
            "python",
            "./notebooks_and_code/functions/xgboost_hyperopt.py",
            "--train_labels",
            f"{base_data_dir}/labels_train_{i}_split.csv/",
            "--train_feats",
            f"{base_data_dir}/features_train_{i}_split.csv/",
            "--valid_labels",
            f"{base_data_dir}/labels_valid_{i}_split.csv/",
            "--valid_feats",
            f"{base_data_dir}/features_valid_{i}_split.csv/",
            "--test_labels",
            f"{base_data_dir}/labels_test_df.csv/",
            "--test_feats",
            f"{base_data_dir}/features_test_df.csv/",
            "--model_directory",
            model_subdir,
            "--epochs",
            "100",
            "--iterations",
            "20",
        ]
    )

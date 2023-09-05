import os
import subprocess

model_dir = "./models/chemprop_rdkit/"
os.makedirs(model_dir, exist_ok=True)

base_data_dir = "./data/metlin_smrt/features/rdkit_csv/cv_splits/"

# Loop through the 5 CV splits
start = 0
end = 4

for i in range(start, end + 1):
    model_subdir = os.path.join(model_dir, f"chemprop_rdkit_{i}/")
    os.makedirs(model_subdir, exist_ok=True)

    # Execute chemprop_hyperopt
    subprocess.run(
        [
            "chemprop_hyperopt",
            "--data_path",
            f"{base_data_dir}/labels_train_{i}_split.csv",
            "--features_path",
            f"{base_data_dir}/features_train_{i}_split.csv",
            "--separate_val_path",
            f"{base_data_dir}/labels_valid_{i}_split.csv",
            "--separate_val_features_path",
            f"{base_data_dir}/features_valid_{i}_split.csv",
            "--no_features_scaling",
            "--dataset_type",
            "regression",
            "--log_dir",
            model_subdir,
            "--config_save_path",
            os.path.join(model_subdir, "config.json"),
            "--hyperopt_checkpoint_dir",
            model_subdir,
            "--metric",
            "mse",
            "--extra_metrics",
            "mae",
            "rmse",
            "r2",
            "--save_preds",
            "--epochs",
            "100",
            "--num_iters",
            "20",
            "--no_cache_mol",
            "--num_workers",
            "20",
        ]
    )

    # Execute chemprop_train
    subprocess.run(
        [
            "chemprop_train",
            "--data_path",
            f"{base_data_dir}/labels_train_{i}_split.csv",
            "--features_path",
            f"{base_data_dir}/features_train_{i}_split.csv",
            "--separate_val_path",
            f"{base_data_dir}/labels_valid_{i}_split.csv",
            "--separate_val_features_path",
            f"{base_data_dir}/features_valid_{i}_split.csv",
            "--separate_test_path",
            f"{base_data_dir}/labels_test_df.csv",
            "--separate_test_features_path",
            f"{base_data_dir}/features_test_df.csv",
            "--no_features_scaling",
            "--dataset_type",
            "regression",
            "--config_path",
            os.path.join(model_subdir, "config.json"),
            "--save_dir",
            model_subdir,
            "--metric",
            "mse",
            "--extra_metrics",
            "mae",
            "rmse",
            "r2",
            "--save_preds",
            "--epochs",
            "100",
            "--no_cache_mol",
            "--num_workers",
            "20",
        ]
    )

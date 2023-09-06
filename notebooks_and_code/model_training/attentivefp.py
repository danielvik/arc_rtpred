import os
import subprocess

model_dir = "./models/attentivefp/"
os.makedirs(model_dir, exist_ok=True)

base_data_dir = "./data/metlin_smrt/features/molgraphconv/cv_splits/"

# Loop through the 5 CV splits
start = 0
end = 4

for i in range(start, end + 1):
    model_subdir = os.path.join(model_dir, f"attentivefp_{i}/")
    os.makedirs(model_subdir, exist_ok=True)

    subprocess.run(
        [
            "python",
            "./notebooks_and_code/functions/attentivefp_hyperopt.py",
            "--train_dir",
            f"{base_data_dir}/train_{i}_split/",
            "--val_dir",
            f"{base_data_dir}/valid_{i}_split/",
            "--test_dir",
            f"{base_data_dir}/test_df/",
            "--model_directory",
            model_subdir,
            "--epochs",
            "100",
            "--callback_intervals",
            "1000",
            "--iterations",
            "20",
        ]
    )

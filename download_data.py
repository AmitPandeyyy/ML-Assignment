"""
download_dataset.py

Downloads the obesity dataset from Kaggle using kagglehub, and moves it to ./data/obesity.csv
"""

import os
import shutil
import kagglehub


TARGET_FILENAME = "ObesityDataSet_raw_and_data_sinthetic.csv"

# search recursively for the target file in the given root directory
def find_file_recursively(root_dir, target_filename):
    for root, dirs, files in os.walk(root_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)

    raise FileNotFoundError(
        f"{target_filename} not found inside {root_dir}"
    )


def download_and_prepare_dataset():
    dataset_path = kagglehub.dataset_download(
        "jayitabhattacharyya/estimation-of-obesity-levels-uci-dataset"
    )
    print("Dataset downloaded to:", dataset_path)

    source_file = find_file_recursively(dataset_path, TARGET_FILENAME)
    print("Found dataset file at:", source_file)

    os.makedirs("data", exist_ok=True)

    destination_file = os.path.join("data", "obesity.csv")

    shutil.copy2(source_file, destination_file)

    print(f"Dataset successfully copied to: {destination_file}")


if __name__ == "__main__":
    download_and_prepare_dataset()

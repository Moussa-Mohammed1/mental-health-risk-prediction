from pathlib import Path
import shutil

import kagglehub


DATASET_HANDLE = "maqdadali/mental-health-risk-prediction-dataset-students"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset() -> Path:
    """Download the Kaggle dataset and copy its files into data/raw."""
    downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(downloaded_path, RAW_DATA_DIR, dirs_exist_ok=True)
    return RAW_DATA_DIR


if __name__ == "__main__":
    destination = download_dataset()
    print(f"Dataset downloaded to {destination}")
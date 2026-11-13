"""Dataset loading helpers."""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/student_mental_health.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def load_dataset(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the student mental-health dataset from a CSV file."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def load_processed_splits(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed train/test features and targets from CSV files."""
    directory = Path(processed_dir)
    paths = {
        "X_train": directory / "X_train_processed.csv",
        "X_test": directory / "X_test_processed.csv",
        "y_train": directory / "y_train.csv",
        "y_test": directory / "y_test.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Processed data file(s) not found: {missing_paths}")

    return (
        pd.read_csv(paths["X_train"]),
        pd.read_csv(paths["X_test"]),
        pd.read_csv(paths["y_train"]).squeeze("columns"),
        pd.read_csv(paths["y_test"]).squeeze("columns"),
    )

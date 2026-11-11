"""Dataset loading helpers."""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/student_mental_health.csv")


def load_dataset(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the student mental-health dataset from a CSV file."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

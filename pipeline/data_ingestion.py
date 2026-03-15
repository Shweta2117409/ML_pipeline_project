"""
data_ingestion.py
-----------------
Responsible for loading raw data from disk into a pandas DataFrame.
"""

import os
import pandas as pd


EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]


def load_data(filepath: str = "data/raw.csv") -> pd.DataFrame:
    """
    Load the raw heart disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the file is empty.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place raw.csv inside the data/ folder."
        )

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"The file at '{filepath}' is empty.")

    print(f"[Ingestion] Loaded {len(df)} rows and {len(df.columns)} columns from '{filepath}'.")
    return df

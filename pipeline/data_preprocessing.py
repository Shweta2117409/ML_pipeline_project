"""
data_preprocessing.py
---------------------
Handles missing value imputation, type coercion, outlier detection (IQR),
and feature/target splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]
TARGET_COLUMN = "target"

# Columns where IQR outlier capping should be applied
CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with the column median."""
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  ✔ Imputed '{col}' with median ({median_val:.2f})")
    return df


def _coerce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns are proper numeric types."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _detect_and_report_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the IQR method and cap them (Winsorization).
    Values below Q1 - 1.5*IQR are capped at the lower bound.
    Values above Q3 + 1.5*IQR are capped at the upper bound.
    """
    print("  [Outlier Detection — IQR Method]")
    for col in CONTINUOUS_COLUMNS:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(
                f"    '{col}': {n_outliers} outlier(s) detected. "
                f"Capped to [{lower:.2f}, {upper:.2f}]."
            )
        else:
            print(f"    '{col}': No outliers found.")
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline:
    1. Coerce numeric types
    2. Impute missing values
    3. Detect and cap outliers (IQR)
    4. Split into features (X) and target (y)

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features), y (target)
    """
    print("[Preprocessing] Starting data preprocessing...")

    df = df.copy()

    # Step 1: Coerce types
    df = _coerce_numeric_types(df)
    print("  ✔ Numeric type coercion complete.")

    # Step 2: Impute missing values
    df = _impute_missing_values(df)

    # Step 3: Outlier detection and capping
    df = _detect_and_report_outliers(df)

    # Step 4: Split features and target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    print(f"[Preprocessing] Complete. Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"  Target distribution — 0: {(y == 0).sum()}, 1: {(y == 1).sum()}")
    return X, y

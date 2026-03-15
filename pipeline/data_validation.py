"""
data_validation.py
------------------
Validates schema, checks for missing values and duplicates,
and ensures column data types are appropriate.
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]

NUMERIC_COLUMNS = REQUIRED_COLUMNS  # All columns in this dataset are numeric


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run validation checks on the raw DataFrame.

    Checks performed:
    - Schema: all required columns are present
    - Missing values: logged (not raised, handled in preprocessing)
    - Duplicate rows: logged and removed
    - Target column: only contains 0 and 1

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ingestion.

    Returns
    -------
    pd.DataFrame
        DataFrame after basic integrity checks (duplicates removed).

    Raises
    ------
    ValueError
        If required columns are missing or target values are invalid.
    """
    print("[Validation] Starting data validation...")

    # --- Schema check ---
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[Validation] FAILED — Missing required columns: {missing_cols}"
        )
    print(f"  ✔ Schema check passed. All {len(REQUIRED_COLUMNS)} required columns present.")

    # --- Missing values check ---
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if not cols_with_missing.empty:
        print(f"  ⚠ Missing values detected (will be imputed in preprocessing):")
        for col, count in cols_with_missing.items():
            print(f"      {col}: {count} missing ({count / len(df):.1%})")
    else:
        print("  ✔ No missing values found.")

    # --- Duplicate rows check ---
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        print(f"  ⚠ Found {n_duplicates} duplicate row(s). Removing them.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("  ✔ No duplicate rows found.")

    # --- Target column check ---
    invalid_targets = df[~df["target"].isin([0, 1, 0.0, 1.0])]
    if not invalid_targets.empty:
        raise ValueError(
            f"[Validation] FAILED — target column contains invalid values: "
            f"{df['target'].unique().tolist()}"
        )
    print("  ✔ Target column contains only valid binary values (0 / 1).")

    # --- Numeric type check ---
    non_numeric = []
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        print(f"  ⚠ Non-numeric columns detected, will coerce: {non_numeric}")
    else:
        print("  ✔ All columns are numeric.")

    print(f"[Validation] Completed. DataFrame shape after validation: {df.shape}")
    return df

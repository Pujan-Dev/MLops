"""Data loading utilities for the IBM Telco Customer Churn dataset.

Responsibilities:
- Download the CSV from IBM's public GitHub mirror if not present locally.
- Load it with pandas and perform minimal cleaning shared by
  training, serving and monitoring code.

Dataset (7043 rows, 21 columns, target = Churn):
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
"""

from pathlib import Path
import urllib.request

import pandas as pd

# Public IBM mirror of the Telco Customer Churn dataset.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

# Project root = parent of src/  ->  track-1/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

# Columns that are numeric after cleaning.
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def download_dataset(dest: Path = DEFAULT_DATA_PATH, url: str = DATA_URL) -> Path:
    """Download the CSV to ``dest`` if it does not exist yet."""
    dest = Path(dest)
    if dest.exists():
        print(f"[data_loader] Dataset already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data_loader] Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[data_loader] Saved -> {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def load_raw(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, downloading it first when missing."""
    path = Path(path)
    if not path.exists():
        download_dataset(path)
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded raw data: {df.shape} from {path}")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal, reproducible cleaning.

    - Strip whitespace from string cells.
    - Coerce ``TotalCharges`` (stored with blanks as strings) to numeric,
      filling the ~11 blank rows with the median.
    - Map target ``Churn`` from Yes/No to 1/0.
    - Drop the ``customerID`` identifier (not a predictive feature).
    """
    df = df.copy()

    # Strip whitespace in all object columns (" No" vs "No" issues).
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA})

    # TotalCharges is object because of blank strings -> numeric.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median_total = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    # Target: Yes/No -> 1/0
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # SeniorCitizen is already 0/1 int; ensure numeric dtypes.
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Identifier carries no signal -> drop.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Drop any residual NaNs (should be none after the steps above).
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) != before:
        print(f"[data_loader] Dropped {before - len(df)} rows with NaNs.")

    return df


def load_clean(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Convenience: download (if needed) + load + clean."""
    return clean_dataframe(load_raw(path))


if __name__ == "__main__":
    df = load_clean()
    print(df.head())
    print(df["Churn"].value_counts(normalize=True))

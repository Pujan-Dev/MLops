"""Preprocessing: train/test split + sklearn ColumnTransformer.

Numeric features    -> median imputation + StandardScaler.
Categorical features -> most-frequent imputation + OneHotEncoder(handle_unknown='ignore').

The same ``build_preprocessor`` is reused by training (``train.py``),
serving (``app.py`` embeds it inside the logged sklearn Pipeline) and
monitoring, so train/serve skew is impossible by construction.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import NUMERIC_FEATURES, TARGET_COLUMN

TEST_SIZE = 0.2
RANDOM_STATE = 42


def infer_feature_types(df: pd.DataFrame, target: str = TARGET_COLUMN) -> Tuple[List[str], List[str]]:
    """Split columns (minus target) into numeric / categorical lists."""
    features = [c for c in df.columns if c != target]
    numeric = [c for c in NUMERIC_FEATURES if c in features]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Return a ColumnTransformer for the given feature lists."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target: str = TARGET_COLUMN,
):
    """Stratified split into X_train, X_test, y_train, y_test + feature lists."""
    numeric_features, categorical_features = infer_feature_types(df, target)
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[preprocess] train={X_train.shape} test={X_test.shape} "
          f"num={len(numeric_features)} cat={len(categorical_features)}")
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

"""
GreenLeaf Grocery - Demand Forecast Preprocessing Pipeline
Task 1: Build data cleaning and preprocessing pipeline

This module provides a reproducible scikit-learn preprocessing pipeline that:
- Loads the raw CSV sales data.
- Splits data into train (70%), validation (15%), and test (15%) splits chronologically.
- Computes imputation statistics strictly on the training split to avoid data leakage.
- Chains transformations using sklearn.pipeline.Pipeline with custom BaseEstimator transformers.
- Imputes missing numerical values (median) and categorical values (most_frequent).
- Encodes categorical fields.
- Logs each transformation with Python logging.
- Saves versioned cleaned CSV files for train, validation, and test.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GreenLeafPreprocess")


def load_raw_data(file_path: str = "data/raw/sales_data.csv") -> pd.DataFrame:
    """Loads raw sales CSV into a pandas DataFrame."""
    if not os.path.exists(file_path):
        logger.error("Raw data file not found at path: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")
    logger.info("Loading raw sales data from: %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Loaded raw data with shape: %s", df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train, validation, and test sets.
    Preserves chronological order if 'date' column is present, otherwise shuffles.
    """
    assert np.isclose(
        train_ratio + val_ratio + test_ratio, 1.0
    ), "Split ratios must sum to 1.0"

    logger.info(
        "Splitting dataset: Train=%.2f, Val=%.2f, Test=%.2f",
        train_ratio,
        val_ratio,
        test_ratio,
    )

    if "date" in df.columns:
        logger.info(
            "Sorting chronologically by 'date' for realistic time-series split."
        )
        df_sorted = df.sort_values("date").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
    else:
        shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = shuffled.iloc[:train_end].copy()
        val_df = shuffled.iloc[train_end:val_end].copy()
        test_df = shuffled.iloc[val_end:].copy()

    logger.info(
        "Split shapes -> Train: %s, Val: %s, Test: %s",
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )
    return train_df, val_df, test_df


class LeakFreeImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that learns imputation statistics strictly during fit()
    (numerical medians and categorical modes) and applies them during transform().
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.statistics_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        stats: Dict[str, Any] = {"numeric_medians": {}, "categorical_modes": {}}

        # Compute medians for numeric columns
        for col in self.numeric_cols:
            if col in X_df.columns:
                median_val = float(X_df[col].dropna().median())
                stats["numeric_medians"][col] = median_val
                logger.info("Computed training median for %s: %s", col, median_val)

        # Compute modes for categorical columns
        for col in self.categorical_cols:
            if col in X_df.columns:
                mode_series = X_df[col].dropna().mode()
                mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                stats["categorical_modes"][col] = mode_val
                logger.info("Computed training mode for %s: %s", col, mode_val)

        self.statistics_ = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        for col, val in self.statistics_.get("numeric_medians", {}).items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(val)

        for col, val in self.statistics_.get("categorical_modes", {}).items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(val)

        return X_df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for one-hot encoding categorical variables
    while preserving clean pandas DataFrame outputs.
    """

    def __init__(self, categorical_cols: Optional[List[str]] = None):
        self.categorical_cols = categorical_cols or []
        self.categories_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        for col in self.categorical_cols:
            if col in X_df.columns:
                cats = sorted(X_df[col].astype(str).unique().tolist())
                self.categories_[col] = cats
                logger.info("Learned categories for %s: %s", col, cats)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        for col, cats in self.categories_.items():
            if col in X_df.columns:
                for cat in cats:
                    dummy_col_name = f"{col}_{cat}"
                    X_df[dummy_col_name] = (X_df[col].astype(str) == cat).astype(int)
        return X_df


def compute_and_apply_imputation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Helper function that computes imputation statistics strictly on the training split,
    and applies them across train, validation, and test splits to prevent data leakage.
    """
    logger.info("Computing imputation statistics on training split...")
    imputer = LeakFreeImputer(
        numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )
    imputer.fit(train_df)

    train_imputed = imputer.transform(train_df)
    val_imputed = imputer.transform(val_df)
    test_imputed = imputer.transform(test_df)

    logger.info("Imputation successfully applied across all splits.")
    return train_imputed, val_imputed, test_imputed, imputer.statistics_


def build_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    """
    Builds an sklearn.pipeline.Pipeline chaining imputation and categorical encoding.
    """
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                LeakFreeImputer(
                    numeric_cols=numeric_cols, categorical_cols=categorical_cols
                ),
            ),
            ("encoder", CategoricalEncoder(categorical_cols=categorical_cols)),
        ]
    )
    return pipeline


def run_pipeline(
    raw_csv_path: str = "data/raw/sales_data.csv", output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the end-to-end data cleaning, imputation, and splitting pipeline.
    Saves clean CSV files in output_dir.
    """
    logger.info("=== Starting GreenLeaf Grocery Data Pipeline ===")

    # 1. Load Data
    df = load_raw_data(raw_csv_path)

    # Define column groups
    numeric_features = ["unit_price", "inventory_level"]
    categorical_features = ["category", "promotion"]

    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    # 2. Split into Train / Validation / Test sets
    train_raw, val_raw, test_raw = split_data(df)

    # 3. Compute and apply imputation (fit on train, transform all)
    train_clean, val_clean, test_clean, stats = compute_and_apply_imputation(
        train_raw, val_raw, test_raw, numeric_features, categorical_features
    )

    # Verify no missing values remain in processed splits
    for name, split in [
        ("Train", train_clean),
        ("Val", val_clean),
        ("Test", test_clean),
    ]:
        null_count = split[numeric_features + categorical_features].isnull().sum().sum()
        logger.info("%s split remaining null values: %d", name, null_count)
        if null_count > 0:
            logger.warning("Unresolved missing values found in %s", name)

    # 4. Save Versioned Clean CSV Files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_clean.to_csv(train_path, index=False)
    val_clean.to_csv(val_path, index=False)
    test_clean.to_csv(test_path, index=False)

    logger.info("Cleaned CSV artifacts successfully saved:")
    logger.info(" - %s (%d rows)", train_path, len(train_clean))
    logger.info(" - %s (%d rows)", val_path, len(val_clean))
    logger.info(" - %s (%d rows)", test_path, len(test_clean))
    logger.info(
        "=== GreenLeaf Grocery Preprocessing Pipeline Finished Successfully ==="
    )

    return train_clean, val_clean, test_clean


if __name__ == "__main__":
    run_pipeline()

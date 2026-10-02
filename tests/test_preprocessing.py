"""
Unit and integration tests for the GreenLeaf Grocery data preprocessing pipeline.
Verifies that:
1. Missing values in numeric and categorical columns are correctly imputed.
2. Imputation statistics are computed strictly on the training set (no data leakage).
3. The scikit-learn Pipeline chains transformations cleanly.
4. The pipeline runs end-to-end and produces valid train, validation, and test CSVs.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess import (  # noqa: E402
    split_data,
    compute_and_apply_imputation,
    build_pipeline,
    run_pipeline,
)
from generate_data import generate_greenleaf_sales_data  # noqa: E402


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic DataFrame with deliberate missing values for testing."""
    data = {
        "date": [
            "2026-06-01",
            "2026-06-02",
            "2026-06-03",
            "2026-06-04",
            "2026-06-05",
            "2026-06-06",
        ],
        "sku_id": ["SKU_001", "SKU_001", "SKU_002", "SKU_002", "SKU_003", "SKU_003"],
        "product_name": [
            "Bananas",
            "Bananas",
            "Spinach",
            "Spinach",
            "Tomatoes",
            "Tomatoes",
        ],
        "category": ["Fruit", "Fruit", "Leafy Greens", None, "Vegetables", "Vegetables"],
        "unit_price": [1.99, np.nan, 3.49, 3.50, np.nan, 2.29],
        "inventory_level": [100, 90, np.nan, 80, 75, 70],
        "promotion": ["No", "Yes", "No", None, "Yes", "No"],
        "is_weekend": [0, 0, 0, 0, 1, 1],
        "units_sold": [50, 60, 30, 35, 45, 40],
    }
    return pd.DataFrame(data)


def test_split_proportions(sample_raw_dataframe):
    """Test that split_data produces the expected partitions and preserves record count."""
    train, val, test = split_data(
        sample_raw_dataframe, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )

    total_rows = len(sample_raw_dataframe)
    assert len(train) + len(val) + len(test) == total_rows
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 2


def test_imputation_removes_all_missing_values(sample_raw_dataframe):
    """Test that compute_and_apply_imputation eliminates all NaN values in target columns."""
    train, val, test = split_data(
        sample_raw_dataframe, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )

    numeric_cols = ["unit_price", "inventory_level"]
    cat_cols = ["category", "promotion"]

    # Verify input has missing values
    assert train["unit_price"].isnull().sum() > 0 or test["unit_price"].isnull().sum() > 0

    train_imp, val_imp, test_imp, stats = compute_and_apply_imputation(
        train, val, test, numeric_cols, cat_cols
    )

    # Verify no missing values in imputed splits
    for col in numeric_cols + cat_cols:
        assert train_imp[col].isnull().sum() == 0, f"Missing values remain in train[{col}]"
        assert val_imp[col].isnull().sum() == 0, f"Missing values remain in val[{col}]"
        assert test_imp[col].isnull().sum() == 0, f"Missing values remain in test[{col}]"


def test_no_data_leakage_in_imputation():
    """
    Test that test set missing values are imputed using TRAIN set median,
    NOT the test set median or global median.
    """
    train_data = pd.DataFrame(
        {"unit_price": [10.0, 10.0, 10.0, 10.0, 10.0]}  # Train median is exactly 10.0
    )
    val_data = pd.DataFrame(
        {"unit_price": [100.0, np.nan]}  # Val missing should receive 10.0 (from train)
    )
    test_data = pd.DataFrame(
        {"unit_price": [500.0, 500.0, np.nan]}  # Test missing should receive 10.0 (from train)
    )

    train_imp, val_imp, test_imp, stats = compute_and_apply_imputation(
        train_data, val_data, test_data, ["unit_price"], []
    )

    # Computed median from train must be 10.0
    assert stats["numeric_medians"]["unit_price"] == 10.0
    # Imputed test value must match train median (10.0), not test values (500.0)
    assert test_imp["unit_price"].iloc[2] == 10.0
    assert val_imp["unit_price"].iloc[1] == 10.0


def test_sklearn_pipeline_chaining():
    """Test that sklearn.pipeline.Pipeline chains imputer and encoder correctly."""
    df = pd.DataFrame(
        {
            "category": ["Fruit", "Vegetables", None],
            "unit_price": [1.99, np.nan, 4.99],
        }
    )
    pipeline = build_pipeline(
        numeric_cols=["unit_price"], categorical_cols=["category"]
    )
    transformed = pipeline.fit_transform(df)

    # Missing unit price and category should be imputed
    assert transformed["unit_price"].isnull().sum() == 0
    assert transformed["category"].isnull().sum() == 0
    # One-hot encoded columns should exist
    assert any("category_" in col for col in transformed.columns)


def test_end_to_end_pipeline(tmp_path):
    """Integration test: Generates raw data, runs pipeline, checks all 3 CSVs exist."""
    raw_path = str(tmp_path / "sales_data.csv")
    out_dir = str(tmp_path / "processed")

    # Generate synthetic raw data
    generate_greenleaf_sales_data(
        start_date="2026-06-01", end_date="2026-06-15", output_path=raw_path
    )
    assert os.path.exists(raw_path)

    # Run preprocessing
    train_df, val_df, test_df = run_pipeline(
        raw_csv_path=raw_path, output_dir=out_dir
    )

    # Check files exist
    assert os.path.exists(os.path.join(out_dir, "train.csv"))
    assert os.path.exists(os.path.join(out_dir, "val.csv"))
    assert os.path.exists(os.path.join(out_dir, "test.csv"))

    # Check non-empty and no null values in imputed fields
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert train_df[["unit_price", "inventory_level"]].isnull().sum().sum() == 0
    assert val_df[["unit_price", "inventory_level"]].isnull().sum().sum() == 0
    assert test_df[["unit_price", "inventory_level"]].isnull().sum().sum() == 0

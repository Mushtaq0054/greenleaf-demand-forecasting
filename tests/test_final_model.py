"""
Unit, integration, and performance tests for Task 3: Tuned Final Model.
Verifies that:
1. Data pipeline correctly engineers time, lag, and rolling features.
2. Model inference produces valid, non-negative predictions of expected shape.
3. Performance test: inference latency is strictly < 100 ms per request.
"""

import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features import FeatureEngineer  # noqa: E402
from train_model import TunedDemandForecaster  # noqa: E402


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic DataFrame for feature pipeline testing."""
    dates = pd.date_range("2026-06-01", periods=10, freq="D")
    records = []
    for d in dates:
        for sku in ["SKU_001", "SKU_002"]:
            records.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "sku_id": sku,
                    "product_name": "Test Product",
                    "category": "Fruit",
                    "unit_price": 2.50,
                    "inventory_level": 70,
                    "promotion": "No",
                    "is_weekend": 1 if d.dayofweek in [5, 6] else 0,
                    "units_sold": 50,
                }
            )
    return pd.DataFrame(records)


def test_data_pipeline_feature_engineering(sample_raw_dataframe):
    """
    Data pipeline test: Verifies that time, lag, and rolling features
    are computed correctly without NaNs.
    """
    fe = FeatureEngineer(lags=[1, 2, 7], rolling_window=7)
    transformed_df = fe.fit_transform(sample_raw_dataframe)

    expected_cols = [
        "day_of_week",
        "month",
        "is_holiday",
        "lag_1_demand",
        "lag_2_demand",
        "lag_7_demand",
        "rolling_7d_mean_demand",
    ]
    for col in expected_cols:
        assert col in transformed_df.columns, f"Missing feature: {col}"
        assert transformed_df[col].isnull().sum() == 0, f"Feature {col} has null values"


def test_model_inference_shape_and_validity():
    """
    Model inference test: Verifies that final_model.joblib can be loaded
    and produces predictions of the expected shape and numeric range.
    """
    model_path = "final_model.joblib"
    assert os.path.exists(model_path), f"Artifact {model_path} does not exist"

    model = joblib.load(model_path)
    assert isinstance(model, TunedDemandForecaster)
    val_df = pd.read_csv("data/processed/val.csv")

    predictions = model.predict(val_df)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(val_df),)
    assert not np.isnan(predictions).any()
    assert (predictions >= 0).all(), "Predictions must not be negative in retail"


def test_inference_latency_performance():
    """
    Performance test: Checks that model inference time is strictly < 100 ms per request.
    Evaluates both single-record latency and average per-request latency.
    """
    model_path = "final_model.joblib"
    model = joblib.load(model_path)
    val_df = pd.read_csv("data/processed/val.csv")

    # Single-record latency benchmark (typical API request simulation)
    single_record = val_df.iloc[[0]].copy()

    # Warmup
    _ = model.predict(single_record)

    latencies = []
    for _ in range(50):
        start_time = time.perf_counter()
        _ = model.predict(single_record)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

    avg_latency_ms = np.mean(latencies)
    max_latency_ms = np.max(latencies)

    # Assert strictly under 100 ms requirement
    assert (
        avg_latency_ms < 100.0
    ), f"Average latency ({avg_latency_ms:.2f} ms) exceeded 100 ms limit"
    assert (
        max_latency_ms < 100.0
    ), f"Peak latency ({max_latency_ms:.2f} ms) exceeded 100 ms limit"

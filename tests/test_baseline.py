"""
Unit and integration tests for Task 2: Baseline Forecasting Model.
Verifies that:
1. The trained baseline model artifact exists and can be loaded with joblib.
2. The model produces valid predictions of the expected shape.
3. Predictions are finite and appropriate for demand forecasting.
4. Evaluation report and plot artifacts exist and are valid.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_baseline import (  # noqa: E402
    BaselineDemandForecaster,
    train_and_evaluate_baseline,
)


@pytest.fixture
def sample_dataset():
    """Provides sample train and validation datasets for testing."""
    train_data = {
        "date": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"],
        "sku_id": ["SKU_001", "SKU_002", "SKU_001", "SKU_002"],
        "product_name": ["Bananas", "Spinach", "Bananas", "Spinach"],
        "category": ["Fruit", "Leafy Greens", "Fruit", "Leafy Greens"],
        "unit_price": [1.99, 3.49, 1.95, 3.50],
        "inventory_level": [100.0, 80.0, 95.0, 85.0],
        "promotion": ["No", "Yes", "No", "No"],
        "is_weekend": [0, 0, 0, 0],
        "units_sold": [85, 45, 90, 42],
    }
    val_data = {
        "date": ["2026-06-05", "2026-06-06"],
        "sku_id": ["SKU_001", "SKU_002"],
        "product_name": ["Bananas", "Spinach"],
        "category": ["Fruit", "Leafy Greens"],
        "unit_price": [2.00, 3.45],
        "inventory_level": [90.0, 75.0],
        "promotion": ["No", "Yes"],
        "is_weekend": [1, 1],
        "units_sold": [110, 58],
    }
    return pd.DataFrame(train_data), pd.DataFrame(val_data)


def test_forecaster_fit_and_predict_shape(sample_dataset):
    """Test that the forecaster fits on training data and produces correct prediction shapes."""
    train_df, val_df = sample_dataset
    forecaster = BaselineDemandForecaster()
    forecaster.fit(train_df)

    predictions = forecaster.predict(val_df)

    # Check predictions are a numpy array with shape equal to val rows
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(val_df),)
    # Check no NaNs in predictions
    assert not np.isnan(predictions).any()


def test_saved_model_can_be_loaded(tmp_path, sample_dataset):
    """Test that saved model can be serialized and deserialized with joblib and used for inference."""
    train_df, val_df = sample_dataset
    model_path = str(tmp_path / "test_baseline_model.joblib")

    forecaster = BaselineDemandForecaster()
    forecaster.fit(train_df)
    forecaster.save(model_path)

    assert os.path.exists(model_path)

    # Load back with joblib
    loaded_model = joblib.load(model_path)
    loaded_predictions = loaded_model.predict(val_df)

    assert isinstance(loaded_predictions, np.ndarray)
    assert loaded_predictions.shape == (len(val_df),)
    assert np.allclose(forecaster.predict(val_df), loaded_predictions)


def test_end_to_end_training_and_artifacts(tmp_path):
    """
    Test the full training and evaluation workflow:
    Produces baseline_model.joblib, baseline_report.md, and evaluation metrics.
    """
    model_path = str(tmp_path / "baseline_model.joblib")
    report_path = str(tmp_path / "baseline_report.md")
    plot_path = str(tmp_path / "predicted_vs_actual.png")

    results = train_and_evaluate_baseline(
        train_path="data/processed/train.csv",
        val_path="data/processed/val.csv",
        model_output_path=model_path,
        report_output_path=report_path,
        plot_output_path=plot_path,
    )

    # Verify metrics exist and are positive numbers
    assert "mae" in results and "rmse" in results
    assert results["mae"] > 0
    assert results["rmse"] > 0
    assert results["rmse"] >= results["mae"]

    # Verify file artifacts were written
    assert os.path.exists(model_path)
    assert os.path.exists(report_path)
    assert os.path.exists(plot_path)

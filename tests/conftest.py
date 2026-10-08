"""
Pytest configuration and session-wide fixtures for GreenLeaf Grocery test suite.
Ensures that data splits (train.csv, val.csv) are automatically generated
in CI environments where raw and processed CSVs are not checked into Git.
"""

import os
import sys
import pytest

# Ensure project root is available on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generate_data import generate_greenleaf_sales_data  # noqa: E402
from preprocess import run_pipeline  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_dataset_splits_exist():
    """
    Session-level auto-use fixture:
    In CI/CD environments (like GitHub Actions runners) where data/ is ignored by .gitignore,
    automatically regenerates sales_data.csv and processed splits (train, val, test)
    so baseline and model evaluation tests never fail with FileNotFoundError.
    """
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "sales_data.csv")
    val_path = os.path.join(PROJECT_ROOT, "data", "processed", "val.csv")
    train_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        if not os.path.exists(raw_path):
            generate_greenleaf_sales_data(
                start_date="2026-06-01",
                end_date="2026-08-31",
                output_path=raw_path,
                random_seed=42,
            )

        run_pipeline(raw_csv_path=raw_path, output_dir=processed_dir)

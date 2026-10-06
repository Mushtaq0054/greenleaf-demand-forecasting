"""
GreenLeaf Grocery - Final Tuned Demand Forecasting Model
Task 3: Feature engineering, model tuning and automated testing

This module imports features from features.py, trains a GradientBoostingRegressor,
performs hyperparameter tuning using RandomizedSearchCV, verifies that the model
achieves at least 10% lower MAE than the baseline, and serializes final_model.joblib.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any

if __name__ == "__main__" and "train_model" not in sys.modules:
    sys.modules["train_model"] = sys.modules["__main__"]

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GreenLeafTuning")


def load_baseline_mae(default_mae: float = 4.0263) -> float:
    """Reads baseline MAE from Task 2 report if available, else returns default."""
    report_path = "baseline_report.md"
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Mean Absolute Error (MAE)" in line:
                        # Extract float value like **4.03 units**
                        parts = line.split("**")
                        for p in parts:
                            clean_str = p.replace("units", "").strip()
                            try:
                                val = float(clean_str)
                                return val
                            except ValueError:
                                continue
        except Exception as e:
            logger.warning("Could not parse baseline MAE from report: %s", e)
    return default_mae


class TunedDemandForecaster(BaseEstimator, RegressorMixin):
    """
    Production Tuned Demand Forecaster for GreenLeaf Grocery.
    Chains feature engineering, categorical encoding, and a tuned GradientBoostingRegressor.
    """

    __module__ = "train_model"

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_window: int = 7,
        n_iter_search: int = 8,
        random_state: int = 42,
    ):
        self.lags = lags or [1, 2, 7]
        self.rolling_window = rolling_window
        self.n_iter_search = n_iter_search
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer(
            lags=self.lags, rolling_window=self.rolling_window
        )
        self.best_estimator_: Optional[GradientBoostingRegressor] = None
        self.feature_names_: List[str] = []
        self.category_levels_: Dict[str, List[str]] = {}

    def _build_feature_matrix(
        self, df: pd.DataFrame, is_training: bool = False
    ) -> pd.DataFrame:
        """Constructs full feature matrix combining engineered time/lag/rolling features."""
        if is_training:
            df_fe = self.feature_engineer.fit_transform(df)
        else:
            df_fe = self.feature_engineer.transform(df)

        numeric_cols = [
            "unit_price",
            "inventory_level",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
            "promo_num",
            "weekend_promo",
            "lag_1_demand",
            "lag_2_demand",
            "lag_7_demand",
            "rolling_7d_mean_demand",
        ]
        cat_cols = ["category", "promotion", "sku_id"]

        df_out = pd.DataFrame(index=df.index)

        # Include numeric features
        for col in numeric_cols:
            if col in df_fe.columns:
                df_out[col] = df_fe[col].astype(float)

        # One-hot encode categorical features
        for col in cat_cols:
            if col in df_fe.columns:
                if is_training:
                    levels = sorted(df_fe[col].astype(str).unique().tolist())
                    self.category_levels_[col] = levels
                else:
                    levels = self.category_levels_.get(col, [])

                for lvl in levels:
                    dummy_name = f"{col}_{lvl}"
                    df_out[dummy_name] = (df_fe[col].astype(str) == lvl).astype(float)

        if is_training:
            self.feature_names_ = df_out.columns.tolist()
        else:
            df_out = df_out.reindex(columns=self.feature_names_, fill_value=0.0)

        return df_out

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Tunes and trains GradientBoostingRegressor using RandomizedSearchCV."""
        logger.info(
            "Starting feature engineering & hyperparameter search on %d records...",
            len(X),
        )

        if y is None:
            if "units_sold" in X.columns:
                y = X["units_sold"].values
            else:
                raise ValueError(
                    "Target column 'units_sold' missing from training data."
                )

        X_train = self._build_feature_matrix(X, is_training=True)

        param_distributions = {
            "n_estimators": [90, 110, 130],
            "learning_rate": [0.05, 0.06, 0.07],
            "max_depth": [3, 4],
            "min_samples_split": [2, 3],
            "subsample": [0.9, 1.0],
        }

        base_gbr = GradientBoostingRegressor(random_state=self.random_state)
        random_search = RandomizedSearchCV(
            estimator=base_gbr,
            param_distributions=param_distributions,
            n_iter=self.n_iter_search,
            scoring="neg_mean_absolute_error",
            cv=3,
            random_state=self.random_state,
            n_jobs=-1,
        )

        random_search.fit(X_train.values, y)
        self.best_estimator_ = random_search.best_estimator_
        logger.info(
            "RandomizedSearchCV best parameters: %s", random_search.best_params_
        )
        logger.info("Best cross-validation MAE: %.4f units", -random_search.best_score_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions, clipping any negative values to 0."""
        if self.best_estimator_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_transformed = self._build_feature_matrix(X, is_training=False)
        preds = self.best_estimator_.predict(X_transformed.values)
        return np.clip(preds, a_min=0.0, a_max=None)

    def save(self, filepath: str = "final_model.joblib") -> None:
        """Serializes model artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Tuned final model artifact serialized to: %s", filepath)


def train_and_evaluate_tuned_model(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    model_output_path: str = "final_model.joblib",
) -> Dict[str, Any]:
    """Orchestrates end-to-end model tuning, evaluation, and serialization."""
    logger.info("=== Starting Task 3: Tuned Model Training & Evaluation ===")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    baseline_mae = load_baseline_mae()
    target_max_mae = baseline_mae * 0.90  # At least 10% lower

    forecaster = TunedDemandForecaster(n_iter_search=8, random_state=42)
    forecaster.fit(train_df)

    # Evaluate on held-out validation set
    y_val = val_df["units_sold"].values
    y_pred = forecaster.predict(val_df)

    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    pct_improvement = ((baseline_mae - mae) / baseline_mae) * 100.0

    logger.info("=== Final Model Performance Comparison ===")
    logger.info("Baseline Model MAE : %.4f units", baseline_mae)
    logger.info("Final Tuned Model MAE: %.4f units", mae)
    logger.info("Final Tuned Model RMSE: %.4f units", rmse)
    logger.info("Improvement over Baseline: %.2f%%", pct_improvement)

    # Verify requirement: at least 10% lower MAE
    if mae > target_max_mae:
        logger.warning(
            "Target 10%% MAE reduction not met (current: %.2f%%, target: >=10%%)",
            pct_improvement,
        )
    else:
        logger.info(
            "SUCCESS: Final model achieved >=10%% lower MAE (%.2f%% improvement)!",
            pct_improvement,
        )

    # Save model artifact
    forecaster.save(model_output_path)

    results = {
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "pct_improvement": pct_improvement,
        "meets_target": mae <= target_max_mae,
    }
    logger.info("=== Task 3 Model Tuning Complete ===")
    return results


if __name__ == "__main__":
    train_and_evaluate_tuned_model()

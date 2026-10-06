"""
GreenLeaf Grocery - Feature Engineering Module
Task 3: Feature engineering, model tuning and automated testing

This module generates time-based features (day of week, month, holidays),
lag features using pandas .shift(), product-level rolling 7-day averages,
and weekend-promotion interaction signals.
"""

import sys
import logging
from typing import Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GreenLeafFeatures")

# Summer retail calendar holidays
KNOWN_HOLIDAYS = {
    "2026-06-19",  # Juneteenth
    "2026-07-04",  # Independence Day
    "2026-07-05",  # Independence Day observed
    "2026-09-07",  # Labor Day
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features: day of week, month, weekend, and holiday flags.
    """
    df_out = df.copy()
    date_dt = pd.to_datetime(df_out["date"])

    df_out["day_of_week"] = date_dt.dt.dayofweek
    df_out["month"] = date_dt.dt.month
    df_out["is_holiday"] = df_out["date"].isin(KNOWN_HOLIDAYS).astype(int)

    # Promo indicator & non-linear synergy
    if "promotion" in df_out.columns:
        promo_num = (df_out["promotion"] == "Yes").astype(float)
        df_out["promo_num"] = promo_num
        if "is_weekend" in df_out.columns:
            df_out["weekend_promo"] = df_out["is_weekend"].astype(float) * promo_num

    return df_out


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    target_col: str = "units_sold",
) -> pd.DataFrame:
    """
    Creates past demand lag features using pandas .shift() grouped by sku_id.
    Preserves original index order to prevent row misalignment.
    """
    df_out = df.copy()
    original_index = df_out.index
    lags = lags or [1, 2, 7]

    df_sorted = df_out.sort_values(["sku_id", "date"])

    for lag in lags:
        col_name = f"lag_{lag}_demand"
        df_sorted[col_name] = (
            df_sorted.groupby("sku_id")[target_col].shift(lag).astype(float)
        )

    # Restore original row index order
    return df_sorted.loc[original_index]


def add_rolling_features(
    df: pd.DataFrame,
    window: int = 7,
    target_col: str = "units_sold",
) -> pd.DataFrame:
    """
    Computes product-level rolling average demand over a specified window (default 7 days).
    Uses .shift(1) before rolling to guarantee today's demand is never leaked into the window.
    Preserves original index order.
    """
    df_out = df.copy()
    original_index = df_out.index
    df_sorted = df_out.sort_values(["sku_id", "date"])

    col_name = f"rolling_{window}d_mean_demand"
    df_sorted[col_name] = (
        df_sorted.groupby("sku_id")[target_col]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        .astype(float)
    )

    return df_sorted.loc[original_index]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies time features,
    lag features (.shift()), and rolling aggregates while imputing initial window NaNs.
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_window: int = 7,
        target_col: str = "units_sold",
    ):
        self.lags = lags or [1, 2, 7]
        self.rolling_window = rolling_window
        self.target_col = target_col
        self.sku_demand_medians_: Dict[str, float] = {}
        self.global_demand_median_: float = 45.0

    def fit(self, X: pd.DataFrame, y=None):
        """Learns default historical medians per SKU for lag imputation."""
        X_df = X.copy()
        if self.target_col in X_df.columns:
            medians = X_df.groupby("sku_id")[self.target_col].median()
            self.sku_demand_medians_ = medians.to_dict()
            self.global_demand_median_ = float(X_df[self.target_col].median())
            logger.info(
                "FeatureEngineer fit: learned SKU medians for %d products.",
                len(self.sku_demand_medians_),
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies all feature transformations and fills initial boundary lag NaNs."""
        X_out = X.copy()

        # 1. Time features
        X_out = add_time_features(X_out)

        # 2. Lag & rolling features
        if self.target_col in X_out.columns:
            X_out = add_lag_features(X_out, lags=self.lags, target_col=self.target_col)
            X_out = add_rolling_features(
                X_out, window=self.rolling_window, target_col=self.target_col
            )
        else:
            # Fallback for inference when target column is absent
            for lag in self.lags:
                X_out[f"lag_{lag}_demand"] = (
                    X_out["sku_id"]
                    .map(self.sku_demand_medians_)
                    .fillna(self.global_demand_median_)
                )
            X_out[f"rolling_{self.rolling_window}d_mean_demand"] = (
                X_out["sku_id"]
                .map(self.sku_demand_medians_)
                .fillna(self.global_demand_median_)
            )

        # Fill boundary lag NaNs using learned SKU baseline medians
        for lag in self.lags:
            lag_col = f"lag_{lag}_demand"
            if lag_col in X_out.columns:
                fallback = (
                    X_out["sku_id"]
                    .map(self.sku_demand_medians_)
                    .fillna(self.global_demand_median_)
                )
                X_out[lag_col] = X_out[lag_col].fillna(fallback)

        roll_col = f"rolling_{self.rolling_window}d_mean_demand"
        if roll_col in X_out.columns:
            fallback = (
                X_out["sku_id"]
                .map(self.sku_demand_medians_)
                .fillna(self.global_demand_median_)
            )
            X_out[roll_col] = X_out[roll_col].fillna(fallback)

        return X_out


def engineer_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolling_window: int = 7,
    target_col: str = "units_sold",
) -> pd.DataFrame:
    """Helper function to engineer all features directly on a DataFrame."""
    fe = FeatureEngineer(
        lags=lags, rolling_window=rolling_window, target_col=target_col
    )
    fe.fit(df)
    return fe.transform(df)

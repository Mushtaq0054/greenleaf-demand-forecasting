"""
GreenLeaf Grocery - Demand Forecasting API Schemas
Pydantic data models for request validation and response serialization.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Input schema for daily SKU demand prediction request.
    Strictly validates business constraints: positive price, non-negative inventory,
    and valid date format.
    """

    date: str = Field(
        ...,
        description="Date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2026-10-15"],
    )
    sku_id: str = Field(
        ...,
        description="Stock Keeping Unit ID (e.g., SKU_001)",
        min_length=1,
        examples=["SKU_001"],
    )
    product_name: str = Field(
        default="Produce Item",
        description="Human-readable product name",
        examples=["Organic Bananas"],
    )
    category: str = Field(
        ...,
        description="Product grocery category",
        examples=["Fruit"],
    )
    unit_price: float = Field(
        ...,
        gt=0.0,
        description="Price per unit in USD. Must be strictly positive (> 0)",
        examples=[2.99],
    )
    inventory_level: int = Field(
        ...,
        ge=0,
        description="Current available stock units. Must be non-negative (>= 0)",
        examples=[85],
    )
    promotion: str = Field(
        default="No",
        description="Promotion active flag ('Yes' or 'No')",
        examples=["No"],
    )

    # Optional historical lag/rolling values (if not provided, pipeline uses learned SKU medians)
    lag_1_demand: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional: Demand from previous day (t-1)",
        examples=[42.0],
    )
    lag_2_demand: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional: Demand from 2 days prior (t-2)",
        examples=[40.0],
    )
    lag_7_demand: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional: Demand from same day previous week (t-7)",
        examples=[45.0],
    )
    rolling_7d_mean_demand: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional: 7-day rolling average demand",
        examples=[43.5],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2026-10-15",
                "sku_id": "SKU_001",
                "product_name": "Organic Honeycrisp Apples",
                "category": "Fruit",
                "unit_price": 2.99,
                "inventory_level": 85,
                "promotion": "No",
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Output schema for demand forecast prediction.
    """

    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    date: str = Field(..., description="Target prediction date")
    predicted_demand: float = Field(
        ..., description="Forecasted daily demand units (rounded, non-negative)"
    )
    unit: str = Field(default="units", description="Measurement unit")
    model_version: str = Field(default="1.0.0", description="Model version")


class HealthResponse(BaseModel):
    """
    Output schema for service health status check.
    """

    status: str = Field(default="healthy", description="Service health state")
    model_loaded: bool = Field(
        default=True, description="Indicates if model artifact is loaded in RAM"
    )
    version: str = Field(default="1.0.0", description="API version")
    model_type: str = Field(
        default="GradientBoostingRegressor",
        description="Underlying machine learning model",
    )

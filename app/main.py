"""
GreenLeaf Grocery - Demand Forecasting FastAPI Service
Task 4: Production REST API microservice for perishable goods forecasting.

Endpoints:
- GET /: Root service metadata
- GET /health: Health check endpoint (checks service & model state)
- POST /predict: Predicts daily demand for a given SKU based on JSON features
- GET /docs: Interactive OpenAPI Swagger documentation
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd

# Ensure root directory is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas import PredictionRequest, PredictionResponse, HealthResponse  # noqa: E402
from app.model_loader import ModelLoader  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GreenLeafAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Pre-loads and caches the ML model into memory
    at application startup, ensuring zero-downtime and avoiding disk I/O on requests.
    """
    logger.info("Initializing GreenLeaf Demand Forecasting API...")
    try:
        # Load and verify model singleton in memory
        model = ModelLoader.get_model()
        logger.info(
            "Model pre-loaded successfully into memory: %s",
            type(model).__name__,
        )
    except Exception as exc:
        logger.error("Failed to pre-load model artifact: %s", exc)
    yield
    logger.info("Shutting down GreenLeaf Demand Forecasting API...")


app = FastAPI(
    title="GreenLeaf Grocery - Demand Forecasting API",
    description=(
        "Production REST microservice providing daily demand forecasts for "
        "perishable grocery goods using a tuned GradientBoosting machine learning model."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get(
    "/",
    tags=["Root"],
    summary="Root Service Information",
    response_description="Returns basic service metadata and documentation links",
)
async def root() -> Dict[str, Any]:
    """Root endpoint returning basic microservice information."""
    return {
        "service": "GreenLeaf Grocery Demand Forecasting API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Service & Model Health Check",
    response_description="Returns health status and whether the ML model is cached in RAM",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Verifies that the microservice is operational and the ML model artifact is loaded.
    """
    model_loaded = ModelLoader.is_loaded()
    if not model_loaded:
        # Attempt lazy load
        try:
            ModelLoader.get_model()
            model_loaded = True
        except Exception:
            model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
        model_type="GradientBoostingRegressor",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Forecasting"],
    summary="Forecast Daily Demand for SKU",
    response_description="Returns forecasted demand units (rounded to 2 decimals)",
)
async def predict_demand(payload: PredictionRequest) -> PredictionResponse:
    """
    Predicts the daily demand for a perishable grocery product based on feature values.

    Validates inputs via Pydantic:
    - Date format (YYYY-MM-DD)
    - Positive unit price (> 0)
    - Non-negative inventory level (>= 0)
    - Valid category & promotion flag
    """
    start_time = time.perf_counter()

    # Convert request payload to DataFrame format expected by FeatureEngineer and Model
    row_dict = {
        "date": payload.date,
        "sku_id": payload.sku_id,
        "product_name": payload.product_name,
        "category": payload.category,
        "unit_price": payload.unit_price,
        "inventory_level": payload.inventory_level,
        "promotion": payload.promotion,
    }

    # Add optional lag/rolling features if provided by client
    if payload.lag_1_demand is not None:
        row_dict["lag_1_demand"] = payload.lag_1_demand
    if payload.lag_2_demand is not None:
        row_dict["lag_2_demand"] = payload.lag_2_demand
    if payload.lag_7_demand is not None:
        row_dict["lag_7_demand"] = payload.lag_7_demand
    if payload.rolling_7d_mean_demand is not None:
        row_dict["rolling_7d_mean_demand"] = payload.rolling_7d_mean_demand

    input_df = pd.DataFrame([row_dict])

    try:
        # Run inference using Singleton ModelLoader
        prediction_val = ModelLoader.predict(input_df)
    except Exception as exc:
        logger.error("Inference failure for SKU %s: %s", payload.sku_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(exc)}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "Forecast generated for SKU %s on %s: %.2f units (latency: %.2f ms)",
        payload.sku_id,
        payload.date,
        prediction_val,
        latency_ms,
    )

    return PredictionResponse(
        sku_id=payload.sku_id,
        date=payload.date,
        predicted_demand=prediction_val,
        unit="units",
        model_version="1.0.0",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

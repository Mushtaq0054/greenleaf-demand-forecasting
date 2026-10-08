"""
Unit, integration, and contract tests for GreenLeaf Demand Forecasting FastAPI Service.
Task 4: Automated API test suite.

Verifies:
1. Root & Health check endpoints return 200 and correct status.
2. Prediction endpoint returns 200 and valid non-negative demand forecast for valid inputs.
3. Strict Pydantic input validation rejects malformed dates, negative prices, and negative inventories with 422.
4. ModelLoader behaves strictly as an in-memory Singleton.
5. OpenAPI documentation schema is properly exposed at /docs and /openapi.json.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import app  # noqa: E402
from app.model_loader import ModelLoader  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance with lifespan startup executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verifies that root endpoint / returns 200 and service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["docs_url"]
    assert "/health" in data["health_url"]
    assert "/predict" in data["predict_url"]


def test_health_endpoint(client):
    """
    Verifies that GET /health returns 200, status is healthy,
    and the model artifact is loaded in RAM.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"
    assert data["model_type"] == "GradientBoostingRegressor"


def test_openapi_schema_accessible(client):
    """
    Verifies OpenAPI schema is accessible and contains required paths: /predict and /health.
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]


def test_predict_valid_payload(client):
    """
    Verifies that POST /predict returns 200 and a valid numeric demand forecast
    for a valid grocery input payload.
    """
    payload = {
        "date": "2026-10-15",
        "sku_id": "SKU_001",
        "product_name": "Organic Honeycrisp Apples",
        "category": "Fruit",
        "unit_price": 2.99,
        "inventory_level": 85,
        "promotion": "No",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU_001"
    assert data["date"] == "2026-10-15"
    assert isinstance(data["predicted_demand"], (int, float))
    assert data["predicted_demand"] >= 0.0
    assert data["unit"] == "units"


def test_predict_with_custom_lags(client):
    """
    Verifies POST /predict works when optional lag values are explicitly passed.
    """
    payload = {
        "date": "2026-10-15",
        "sku_id": "SKU_002",
        "product_name": "Fresh Organic Spinach",
        "category": "Vegetable",
        "unit_price": 3.49,
        "inventory_level": 50,
        "promotion": "Yes",
        "lag_1_demand": 45.0,
        "lag_2_demand": 48.0,
        "lag_7_demand": 50.0,
        "rolling_7d_mean_demand": 46.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_demand"] >= 0.0


def test_predict_validation_missing_required_field(client):
    """
    Verifies that omitting a required field (e.g., unit_price) returns HTTP 422.
    """
    invalid_payload = {
        "date": "2026-10-15",
        "sku_id": "SKU_001",
        # Missing 'unit_price'
        "category": "Fruit",
        "inventory_level": 85,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_negative_unit_price(client):
    """
    Verifies that non-positive unit_price (< 0) is rejected with HTTP 422.
    """
    invalid_payload = {
        "date": "2026-10-15",
        "sku_id": "SKU_001",
        "product_name": "Invalid Item",
        "category": "Fruit",
        "unit_price": -2.50,  # Negative price is invalid
        "inventory_level": 85,
        "promotion": "No",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_negative_inventory(client):
    """
    Verifies that negative inventory (< 0) is rejected with HTTP 422.
    """
    invalid_payload = {
        "date": "2026-10-15",
        "sku_id": "SKU_001",
        "category": "Fruit",
        "unit_price": 2.50,
        "inventory_level": -5,  # Negative inventory is invalid
        "promotion": "No",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_invalid_date_format(client):
    """
    Verifies that invalid date format (not YYYY-MM-DD) is rejected with HTTP 422.
    """
    invalid_payload = {
        "date": "15/10/2026",  # Invalid format
        "sku_id": "SKU_001",
        "category": "Fruit",
        "unit_price": 2.50,
        "inventory_level": 85,
        "promotion": "No",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_singleton_model_caching():
    """
    Verifies that ModelLoader strictly returns the same cached instance in memory
    and does not reload the model on successive calls.
    """
    model_instance_1 = ModelLoader.get_model()
    model_instance_2 = ModelLoader.get_model()
    assert model_instance_1 is model_instance_2
    assert id(model_instance_1) == id(model_instance_2)

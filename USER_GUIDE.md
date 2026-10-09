# GreenLeaf Grocery - Demand Forecasting API User Guide

Welcome to the **GreenLeaf Grocery Demand Forecasting Service** user manual. This document guides store managers, inventory personnel, and software integrations on querying the daily demand prediction API.

---

## 1. Service Overview & Architecture

The forecasting service is deployed as a high-performance REST microservice. It allows store managers to send daily inventory and pricing parameters for perishable produce items and receive immediate, machine-learning-backed sales demand forecasts.

### Service Base URLs
- **Local Development URL:** `http://localhost:8000`
- **Interactive Swagger Documentation:** `http://localhost:8000/docs` (or `<LIVE_URL>/docs`)
- **Live Production URL:** Listed in project [`README.md`](README.md) upon Render deployment.

---

## 2. Authentication

- **Current Release (v1.0.0):** Standard open endpoints within the internal grocery store network. No API key or bearer token is required for internal queries to `/health` and `/predict`.
- **Future Integration:** For external multi-store networks, API key headers (`X-API-Key: <token>`) can be configured via environment variables.

---

## 3. Available API Endpoints

| Method | Endpoint | Description | Expected Status |
|---|---|---|---|
| `GET` | `/` | Root service metadata and navigation links | `200 OK` |
| `GET` | `/health` | Service and ML model health check status | `200 OK` |
| `POST` | `/predict` | Primary daily demand prediction endpoint | `200 OK` |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI for browser testing | `200 OK` |

---

## 4. Querying the Prediction Endpoint (`POST /predict`)

### 4.1 Input JSON Schema (Request Payload)

| Field Name | Type | Required? | Constraints & Validation | Example |
|---|---|---|---|---|
| `date` | String | **Yes** | Standard `YYYY-MM-DD` regex format | `"2026-10-15"` |
| `sku_id` | String | **Yes** | Valid product identifier | `"SKU_001"` |
| `product_name` | String | No | Descriptive produce title | `"Organic Bananas"` |
| `category` | String | **Yes** | Produce category (`Fruit`, `Vegetables`, `Leafy Greens`, `Berries`) | `"Fruit"` |
| `unit_price` | Float | **Yes** | Must be strictly positive (`> 0.0`) | `2.99` |
| `inventory_level`| Integer| **Yes** | Current stock on hand (`>= 0`) | `85` |
| `promotion` | String | No | Promotional status: `"Yes"` or `"No"` | `"No"` |
| `lag_1_demand` | Float | No | Yesterday's demand (optional signal) | `42.0` |
| `lag_2_demand` | Float | No | Demand 2 days ago (optional signal) | `40.0` |
| `lag_7_demand` | Float | No | Demand same day last week (optional signal) | `45.0` |
| `rolling_7d_mean_demand` | Float | No | 7-day average sales (optional signal) | `43.5` |

> 💡 **Note on Optional Signals:** If lag and rolling fields are omitted, the service automatically utilizes historical learned SKU medians from the training pipeline.

---

### 4.2 Output JSON Schema (Response Payload)

```json
{
  "sku_id": "SKU_001",
  "date": "2026-10-15",
  "predicted_demand": 41.28,
  "unit": "units",
  "model_version": "1.0.0"
}
```

- `predicted_demand`: Expected quantity of units to be sold on target date (rounded to 2 decimal places, non-negative).
- `unit`: Quantity unit (always `"units"`).

---

## 5. Ready-to-Use `curl` Commands

### Example 1: Verify Service Health
```bash
curl -X GET "http://localhost:8000/health" \
     -H "accept: application/json"
```

**Expected Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "model_type": "GradientBoostingRegressor"
}
```

---

### Example 2: Predict Demand for Standard Produce Item (Apples)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2026-10-15",
       "sku_id": "SKU_001",
       "product_name": "Organic Honeycrisp Apples",
       "category": "Fruit",
       "unit_price": 2.99,
       "inventory_level": 85,
       "promotion": "No"
     }'
```

---

### Example 3: Predict Demand with Promotional Signal & Custom Lags
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2026-10-15",
       "sku_id": "SKU_002",
       "product_name": "Organic Baby Spinach",
       "category": "Vegetables",
       "unit_price": 3.49,
       "inventory_level": 50,
       "promotion": "Yes",
       "lag_1_demand": 48.0,
       "lag_2_demand": 45.0,
       "lag_7_demand": 52.0,
       "rolling_7d_mean_demand": 47.1
     }'
```

---

## 6. Python Integration Example

Store automation scripts can integrate with the API using standard Python:

```python
import requests

API_URL = "http://localhost:8000/predict"

payload = {
    "date": "2026-10-15",
    "sku_id": "SKU_001",
    "product_name": "Organic Bananas",
    "category": "Fruit",
    "unit_price": 1.99,
    "inventory_level": 120,
    "promotion": "No"
}

response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    data = response.json()
    forecast = data["predicted_demand"]
    print(f"Recommended order replenishment for {payload['sku_id']}: {forecast} units.")
else:
    print(f"API Error {response.status_code}: {response.text}")
```

---

## 7. Error Codes & Troubleshooting

| HTTP Status | Meaning | Cause | Action Required |
|---|---|---|---|
| `200 OK` | Success | Valid input payload | Use `predicted_demand` for ordering |
| `422 Unprocessable Entity` | Validation Error | Negative price, negative stock, or wrong date format | Inspect response details; ensure `unit_price > 0`, `date` is `YYYY-MM-DD` |
| `500 Internal Server Error` | Model Failure | Unexpected server issue or missing artifact | Check `/health` endpoint to verify model is loaded |
| `503 Service Unavailable` | Cold Start | Cloud instance warming up | Retry request in 10-15 seconds |

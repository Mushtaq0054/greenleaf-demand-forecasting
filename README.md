# Perishable Goods Demand Forecast for GreenLeaf Grocery

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD%20Pipeline-2088FF?logo=githubactions&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting%20%26%20Pipelines-orange?logo=scikitlearn)
![pytest](https://img.shields.io/badge/pytest-19%20passed-brightgreen?logo=pytest)
![Performance](https://img.shields.io/badge/Inference%20Latency-%3C%203ms%20(SLA%20%3C%20100ms)-blue)
![Tasks](https://img.shields.io/badge/Ezitech%20Internship-Tasks%201%2C%202%2C%203%20%26%204%20Complete-green)

**Client:** GreenLeaf Grocery – Neighborhood Organic Market  
**Intern:** Mushtaq Ahmad  
**Repository:** [https://github.com/Mushtaq0054/greenleaf-demand-forecasting](https://github.com/Mushtaq0054/greenleaf-demand-forecasting)

---

## 1. Project Overview & Business Context

GreenLeaf Grocery is an organic neighborhood market stocking fresh, perishable produce. Because fresh produce spoils quickly, inaccurate demand forecasting causes excessive inventory waste and lost revenue.

This repository implements the complete end-to-end forecasting solution:
- **Task 1: Data Cleaning & Preprocessing Pipeline** (Reproducible scikit-learn pipeline, leak-free imputation, train/val/test splits, versioned CSV artifacts).
- **Task 2: Develop Baseline Forecasting Model & Report** (Linear Regression baseline, StandardScaler numeric scaling, validation evaluation with MAE/RMSE, visualization, serialized joblib model, and evaluation report).
- **Task 3: Feature Engineering, Model Tuning & Automated Testing** (Calendar/lag/rolling feature engineering, tuned GradientBoostingRegressor with RandomizedSearchCV, achieving **14.82% lower MAE** than baseline, latency benchmark $<100\text{ ms}$, comprehensive `TEST_PLAN.md`, and automated pytest suite).
- **Task 4: Production FastAPI Service, Docker Containerization & CI/CD** (Production REST API, Singleton model caching in RAM, strict Pydantic v2 input validation, containerized with Docker, automated CI pipeline with GitHub Actions).

---

## 2. System Architecture

```text
                                [ Client / Store System ]
                                            │
                                            ▼
                               [ POST /predict (JSON) ]
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │                                  Docker Container (:8000)                       │
   │                                                                                 │
   │   FastAPI Gateway                                                               │
   │   ├── GET  /health ──────────► Checks Container & In-Memory Model State         │
   │   ├── GET  /docs   ──────────► Interactive OpenAPI / Swagger UI                 │
   │   └── POST /predict                                                             │
   │            │                                                                    │
   │            ▼                                                                    │
   │   Pydantic Schema Validation (Price > 0, Inventory >= 0, Date Regex)            │
   │            │                                                                    │
   │            ▼                                                                    │
   │   ModelLoader (Thread-safe Singleton in RAM)                                    │
   │            │                                                                    │
   │            ▼                                                                    │
   │   TunedDemandForecaster (final_model.joblib)                                    │
   │   ├── Feature Transformation (Calendar, Lags [1, 2, 7], Rolling 7-day Mean)     │
   │   └── Tuned GradientBoostingRegressor                                           │
   │            │                                                                    │
   │            ▼                                                                    │
   │   Response: {"predicted_demand": 42.80, "sku_id": "SKU_001", "unit": "units"}   │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Performance Comparison

| Metric | Task 2 Baseline (Linear Regression) | Task 3 Final Model (GradientBoosting) | Improvement | Target Requirement |
|---|---|---|---|---|
| **Validation MAE** | `4.0300 units` | **`3.4329 units`** | **14.82% reduction** | $\ge 10.0\%$ reduction |
| **Validation RMSE** | `5.3254 units` | **`4.9158 units`** | **7.69% reduction** | Lower is better |
| **Cross-Validation MAE** | N/A | `3.8233 units` | Robust generalization | CV stability |
| **Inference Latency** | $< 1\text{ ms}$ | **$\approx 2.4\text{ ms}$** | Ultra-responsive | $< 100\text{ ms}$ SLA |

---

## 4. Repository File Structure

```text
greenleaf-demand-forecasting/
├── app/
│   ├── __init__.py                  # Application package marker
│   ├── main.py                      # FastAPI application with /health, /predict, /docs
│   ├── schemas.py                   # Pydantic v2 data models with validation
│   └── model_loader.py              # Thread-safe Singleton model loader in RAM
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI workflow (lint -> test -> build)
├── data/
│   ├── raw/
│   │   ├── .gitkeep                 # Retains directory structure in Git
│   │   └── sales_data.csv           # Raw sales records (protected via .gitignore)
│   └── processed/
│       ├── .gitkeep                 # Retains directory structure in Git
│       ├── train.csv                # 70% cleaned training set (protected)
│       ├── val.csv                  # 15% cleaned validation set (protected)
│       └── test.csv                 # 15% cleaned test set (protected)
├── reports/
│   └── figures/
│       └── predicted_vs_actual.png  # Task 2 actual vs predicted demand visualization
├── tests/
│   ├── test_preprocessing.py        # Task 1 unit tests (5 tests)
│   ├── test_baseline.py             # Task 2 unit tests (3 tests)
│   ├── test_final_model.py          # Task 3 unit & latency benchmark tests (3 tests)
│   └── test_api.py                  # Task 4 FastAPI unit, contract & singleton tests (8 tests)
├── Dockerfile                       # Production Python 3.11-slim container image
├── .dockerignore                    # Optimization rules excluding virtualenvs & git
├── baseline_model.joblib            # Serialized baseline model artifact
├── final_model.joblib               # Serialized final tuned GradientBoosting model
├── baseline_report.md               # Task 2 formal evaluation report
├── TEST_PLAN.md                     # Task 3 formal test plan & verification matrix
├── generate_data.py                 # Raw data generation script
├── preprocess.py                    # Scikit-learn Pipeline with leak-free imputer
├── features.py                      # Modular feature engineering module (time, lag, rolling)
├── train_baseline.py                # Baseline training & evaluation script
├── train_model.py                   # Final model hyperparameter search & training
├── requirements.txt                 # Clean production & development dependencies
├── .gitignore                       # Protection rules for data files and artifacts
└── README.md                        # Complete project documentation
```

---

## 5. Quickstart & Local Execution

### Step 1: Clone the repository
```bash
git clone https://github.com/Mushtaq0054/greenleaf-demand-forecasting.git
cd greenleaf-demand-forecasting
```

### Step 2: Set up Virtual Environment & Install Dependencies
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run FastAPI Service Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Service Root:** `http://localhost:8000/`
- **Health Check:** `http://localhost:8000/health`
- **Interactive OpenAPI Documentation:** `http://localhost:8000/docs`

---

## 6. Docker Containerization (Task 4)

The application is fully containerized using a secure, lightweight `python:3.11-slim` base image with built-in health-checks.

### Build the Docker Image
```bash
docker build -t greenleaf-forecast:latest .
```

### Run the Container Locally
```bash
docker run -d -p 8000:8000 --name forecast-service greenleaf-forecast:latest
```

### Verify Container Health
```bash
docker ps
curl http://localhost:8000/health
```

### View Logs & Stop Container
```bash
# View real-time logs
docker logs -f forecast-service

# Stop and remove container
docker stop forecast-service
docker rm forecast-service
```

---

## 7. API Usage & Testing with `curl`

### 1. Health Check Endpoint (`GET /health`)
```bash
curl -X GET "http://localhost:8000/health" -H "accept: application/json"
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

### 2. Demand Forecast Endpoint (`POST /predict`)
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

**Expected Response (`200 OK`):**
```json
{
  "sku_id": "SKU_001",
  "date": "2026-10-15",
  "predicted_demand": 41.28,
  "unit": "units",
  "model_version": "1.0.0"
}
```

---

### 3. Prediction with Custom Lag/Rolling Signals
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2026-10-15",
       "sku_id": "SKU_002",
       "product_name": "Fresh Organic Spinach",
       "category": "Vegetables",
       "unit_price": 3.49,
       "inventory_level": 50,
       "promotion": "Yes",
       "lag_1_demand": 45.0,
       "lag_2_demand": 48.0,
       "lag_7_demand": 50.0,
       "rolling_7d_mean_demand": 46.2
     }'
```

---

### 4. Validation Error Handling (`422 Unprocessable Entity`)
If negative price, negative inventory, or malformed date is supplied:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "date": "2026-10-15",
       "sku_id": "SKU_001",
       "unit_price": -5.00,
       "inventory_level": -10,
       "category": "Fruit"
     }'
```
Returns structured HTTP 422 explaining the exact constraint violation.

---

## 8. Continuous Integration & Deployment (CI/CD)

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every `push` and `pull_request` to `main`:

```text
GitHub Push
     │
     ▼
[ Job: lint-test-build ]
  ├── 1. Checkout repository & setup Python 3.11
  ├── 2. Install dependencies (pip install -r requirements.txt)
  ├── 3. Code linting (flake8 syntax & style verification)
  ├── 4. Run automated test suite (pytest -v)
  ├── 5. Build Docker container image (docker build)
  ├── 6. Smoke test container locally (verify GET /health returns 200)
  └── 7. (Optional) Push image to Docker Hub with GitHub Secrets
```

---

## 9. Automated Testing (19 Tests Passing)

### Running pytest
```bash
pytest --verbose
```

### Test Suite Overview:
- **`tests/test_preprocessing.py` (5 tests):** Split proportions, leak-free imputation, scikit-learn pipeline chaining.
- **`tests/test_baseline.py` (3 tests):** Baseline linear regression fitting, evaluation metrics, artifact serialization.
- **`tests/test_final_model.py` (3 tests):** Time/lag/rolling feature generation, inference shape, latency benchmark ($< 100\text{ ms}$ SLA).
- **`tests/test_api.py` (8 tests):** Health check, valid prediction, custom lags, negative price rejection, negative inventory rejection, date format validation, Singleton model identity check, and OpenAPI schema verification.

---

## 10. Mentor & Supervisor Submission Notes

### Task 1 Note:
> "Implemented reproducible data cleaning pipeline with custom leak-free imputer in `preprocess.py`, zero null values in processed splits, protected data via `.gitignore`, and passing unit tests."

### Task 2 Note:
> "Trained Multiple Linear Regression baseline model with standardized numerical features (`StandardScaler`) and one-hot encoded categories. Serialized model to `baseline_model.joblib`. Achieved validation MAE of 4.0300 units and RMSE of 5.3254 units. Authored formal evaluation report in `baseline_report.md` with comparison plot in `reports/figures/predicted_vs_actual.png`. All pytest test cases pass and code conforms to PEP 8 standards."

### Task 3 Note:
> "Engineered temporal, lag (1, 2, 7 days), and 7-day rolling features in modular `features.py`. Tuned GradientBoostingRegressor in `train_model.py` using RandomizedSearchCV. The tuned final model achieved validation MAE of 3.4329 units (14.82% lower error than baseline, surpassing the >=10% requirement) and is serialized in `final_model.joblib`. Implemented automated pytest suite covering data pipelines, model inference, and latency performance (< 3 ms vs. < 100 ms SLA). Documented formal test plan in `TEST_PLAN.md`."

### Task 4 Note:
> "Operationalized the final demand forecasting model into a production-grade FastAPI microservice in `app/main.py`. Implemented thread-safe Singleton caching in `app/model_loader.py` to keep the model in memory across requests, and strict Pydantic v2 input validation in `app/schemas.py`. Created endpoints for service health (`GET /health`) and real-time inference (`POST /predict`), with interactive OpenAPI documentation at `/docs`. Containerized the application with an optimized `Dockerfile` (Python 3.11-slim) and `.dockerignore`. Configured an automated GitHub Actions CI workflow in `.github/workflows/ci.yml` that lints code with flake8, runs all 19 unit & API tests with pytest, builds the Docker image, and verifies health status via a local smoke test."

# Perishable Goods Demand Forecast for GreenLeaf Grocery

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD%20%26%20Monitoring-2088FF?logo=githubactions&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting%20%26%20Pipelines-orange?logo=scikitlearn)
![pytest](https://img.shields.io/badge/pytest-19%20passed-brightgreen?logo=pytest)
![Performance](https://img.shields.io/badge/Inference%20Latency-%3C%203ms%20(SLA%20%3C%20100ms)-blue)
![Tasks](https://img.shields.io/badge/Ezitech%20Internship-All%205%20Tasks%20Complete-brightgreen)

**Client:** GreenLeaf Grocery – Neighborhood Organic Market  
**Intern:** Mushtaq Ahmad  
**Repository:** [https://github.com/Mushtaq0054/greenleaf-demand-forecasting](https://github.com/Mushtaq0054/greenleaf-demand-forecasting)  
**Live Web Dashboard:** [https://greenleaf-demand-forecasting.onrender.com/dashboard](https://greenleaf-demand-forecasting.onrender.com/dashboard) *(Interactive Grocery UI)*  
**Live Production API:** [https://greenleaf-demand-forecasting.onrender.com](https://greenleaf-demand-forecasting.onrender.com) *(Render Web Service)*  
**Live Interactive Docs:** [https://greenleaf-demand-forecasting.onrender.com/docs](https://greenleaf-demand-forecasting.onrender.com/docs)  

---

## 1. Project Overview & Business Context

GreenLeaf Grocery is an organic neighborhood market stocking fresh, perishable produce. Because fresh produce spoils quickly, inaccurate demand forecasting causes excessive inventory waste and lost revenue.

This repository implements the complete end-to-end forecasting solution across all 5 milestones:
- **Task 1: Data Cleaning & Preprocessing Pipeline** (Reproducible scikit-learn pipeline, leak-free imputation, train/val/test splits, versioned CSV artifacts).
- **Task 2: Develop Baseline Forecasting Model & Report** (Linear Regression baseline, StandardScaler numeric scaling, validation evaluation with MAE/RMSE, visualization, serialized joblib model, and evaluation report).
- **Task 3: Feature Engineering, Model Tuning & Automated Testing** (Calendar/lag/rolling feature engineering, tuned GradientBoostingRegressor with RandomizedSearchCV, achieving **14.82% lower MAE** than baseline, latency benchmark $<100\text{ ms}$, comprehensive `TEST_PLAN.md`, and automated pytest suite).
- **Task 4: Production FastAPI Service, Docker Containerization & CI/CD** (Production REST API, Singleton model caching in RAM, strict Pydantic v2 input validation, containerized with Docker, automated CI pipeline with GitHub Actions).
- **Task 5: Model Card, User Guide, Cloud Deployment & Latency Monitoring** (Comprehensive [`MODEL_CARD.md`](MODEL_CARD.md), staff manual [`USER_GUIDE.md`](USER_GUIDE.md), live deployment on Render, and scheduled daily latency monitoring script `monitor.py`).

---

## 2. Key Documentation Links

- 📄 **[Model Card (MODEL_CARD.md)](MODEL_CARD.md):** Standardized ML model documentation detailing training data, performance metrics, limitations, environmental benefits, and fairness considerations.
- 📘 **[User Guide (USER_GUIDE.md)](USER_GUIDE.md):** Operational manual for store managers and technical staff explaining query formats, authentication, and `curl`/Python examples.
- 📋 **[Test Plan (TEST_PLAN.md)](TEST_PLAN.md):** Full automated test specification and SLA latency benchmark results.
- 📊 **[Baseline Evaluation Report (baseline_report.md)](baseline_report.md):** Formal comparison report between baseline Linear Regression and tuned GradientBoosting models.

---

## 3. System Architecture

```text
                                [ Client / Store System ]
                                            │
                                            ▼
                               [ POST /predict (JSON) ]
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   │                       Docker Container on Render Cloud                          │
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

## 4. Model Performance Comparison

| Metric | Task 2 Baseline (Linear Regression) | Task 3 Final Model (GradientBoosting) | Improvement | Target Requirement |
|---|---|---|---|---|
| **Validation MAE** | `4.0300 units` | **`3.4329 units`** | **14.82% reduction** | $\ge 10.0\%$ reduction |
| **Validation RMSE** | `5.3254 units` | **`4.9158 units`** | **7.69% reduction** | Lower is better |
| **Cross-Validation MAE** | N/A | `3.8233 units` | Robust generalization | CV stability |
| **Inference Latency** | $< 1\text{ ms}$ | **$\approx 2.4\text{ ms}$** | Ultra-responsive | $< 100\text{ ms}$ SLA |

---

## 5. Repository File Structure

```text
greenleaf-demand-forecasting/
├── app/
│   ├── __init__.py                  # Application package marker
│   ├── main.py                      # FastAPI application with /health, /predict, /docs
│   ├── schemas.py                   # Pydantic v2 data models with validation
│   └── model_loader.py              # Thread-safe Singleton model loader in RAM
├── .github/
│   └── workflows/
│       ├── ci.yml                   # CI/CD workflow (lint -> test -> docker build)
│       └── monitor.yml              # Scheduled daily latency monitoring workflow
├── data/
│   ├── raw/                         # Raw sales data (protected via .gitignore)
│   └── processed/                   # Cleaned train, val, test splits (protected)
├── reports/
│   ├── figures/
│   │   └── predicted_vs_actual.png  # Actual vs predicted demand visualization
│   └── monitoring_logs.csv          # Daily latency monitoring log output
├── tests/
│   ├── conftest.py                  # Pytest fixtures and dynamic data generation
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
├── MODEL_CARD.md                    # Task 5 standardized model card
├── USER_GUIDE.md                    # Task 5 store staff user manual
├── monitor.py                       # Task 5 daily service latency monitoring script
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

## 6. Cloud Deployment on Render (Task 5)

The service is configured for direct automated deployment on [Render](https://render.com) using the project's `Dockerfile`:

### Deployment Steps:
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** ➔ **Web Service**.
3. Connect your GitHub repository `https://github.com/Mushtaq0054/greenleaf-demand-forecasting`.
4. Configure service settings:
   - **Environment:** `Docker`
   - **Region:** Any preferred (e.g., Oregon / Frankfurt)
   - **Branch:** `main`
   - **Plan:** `Free`
5. Click **Create Web Service**. Render builds the image from the `Dockerfile` and assigns a secure live HTTPS domain.

---

## 7. Daily Latency & Health Monitoring (Task 5)

The repository includes an automated monitoring script ([`monitor.py`](monitor.py)) and a scheduled GitHub Actions cron workflow ([`.github/workflows/monitor.yml`](.github/workflows/monitor.yml)):

- **Schedule:** Runs automatically every day at 06:00 UTC (11:00 AM PKT).
- **Execution:** Sends health and prediction requests to the service, records response latency in milliseconds, and appends records to `reports/monitoring_logs.csv`.
- **Manual Trigger:** Can also be triggered on-demand via the **Actions** tab in GitHub by selecting **Daily Service Latency Monitoring** ➔ **Run workflow**.

---

## 8. API Usage with `curl`

### Health Check Endpoint
```bash
curl -X GET "https://greenleaf-demand-forecasting.onrender.com/health" -H "accept: application/json"
```

### Demand Prediction Endpoint
```bash
curl -X POST "https://greenleaf-demand-forecasting.onrender.com/predict" \
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

**Response (`200 OK`):**
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

## 9. Mentor & Supervisor Submission Notes

### Task 1 Note:
> "Implemented reproducible data cleaning pipeline with custom leak-free imputer in `preprocess.py`, zero null values in processed splits, protected data via `.gitignore`, and passing unit tests."

### Task 2 Note:
> "Trained Multiple Linear Regression baseline model with standardized numerical features (`StandardScaler`) and one-hot encoded categories. Serialized model to `baseline_model.joblib`. Achieved validation MAE of 4.0300 units and RMSE of 5.3254 units. Authored formal evaluation report in `baseline_report.md` with comparison plot in `reports/figures/predicted_vs_actual.png`. All pytest test cases pass and code conforms to PEP 8 standards."

### Task 3 Note:
> "Engineered temporal, lag (1, 2, 7 days), and 7-day rolling features in modular `features.py`. Tuned GradientBoostingRegressor in `train_model.py` using RandomizedSearchCV. The tuned final model achieved validation MAE of 3.4329 units (14.82% lower error than baseline, surpassing the >=10% requirement) and is serialized in `final_model.joblib`. Implemented automated pytest suite covering data pipelines, model inference, and latency performance (< 3 ms vs. < 100 ms SLA). Documented formal test plan in `TEST_PLAN.md`."

### Task 4 Note:
> "Operationalized the final demand forecasting model into a production-grade FastAPI microservice in `app/main.py`. Implemented thread-safe Singleton caching in `app/model_loader.py` to keep the model in memory across requests, and strict Pydantic v2 input validation in `app/schemas.py`. Created endpoints for service health (`GET /health`) and real-time inference (`POST /predict`), with interactive OpenAPI documentation at `/docs`. Containerized the application with an optimized `Dockerfile` (Python 3.11-slim) and `.dockerignore`. Configured an automated GitHub Actions CI workflow in `.github/workflows/ci.yml` that lints code with flake8, runs all 19 unit & API tests with pytest, builds the Docker image, and verifies health status via a local smoke test."

### Task 5 Note:
> "Authored comprehensive model documentation in MODEL_CARD.md following industry standards (covering data provenance, 14.82% MAE improvement, limitations, and food waste reduction ethics). Developed a clear store staff operational manual in USER_GUIDE.md detailing request schemas, authentication, and curl/Python examples. Configured cloud container deployment for Render web services and implemented an automated daily latency monitoring script in monitor.py orchestrated via a scheduled GitHub Actions cron workflow (.github/workflows/monitor.yml) that logs daily request latency."

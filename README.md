# Perishable Goods Demand Forecast for GreenLeaf Grocery

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting%20%26%20Pipelines-orange?logo=scikitlearn)
![pytest](https://img.shields.io/badge/pytest-11%20passed-brightgreen?logo=pytest)
![Performance](https://img.shields.io/badge/Inference%20Latency-%3C%203ms%20(SLA%20%3C%20100ms)-blue)
![Code Style](https://img.shields.io/badge/code%20style-black%20%26%20flake8-000000.svg)
![Data Versioning](https://img.shields.io/badge/Data%20Versioning-.gitignore%20Protected-success)
![Tasks](https://img.shields.io/badge/Ezitech%20Internship-Tasks%201%2C%202%20%26%203%20Complete-green)

**Client:** GreenLeaf Grocery – Neighborhood Organic Market  
**Intern:** Mushtaq Ahmad  
**Repository:** [https://github.com/Mushtaq0054/greenleaf-demand-forecasting](https://github.com/Mushtaq0054/greenleaf-demand-forecasting)

---

## 1. Project Overview & Business Context

GreenLeaf Grocery is an organic neighborhood market stocking fresh, perishable produce. Because fresh produce spoils quickly, inaccurate demand forecasting causes excessive inventory waste and lost revenue.

This repository implements the end-to-end forecasting pipeline:
- **Task 1: Data Cleaning & Preprocessing Pipeline** (Reproducible scikit-learn pipeline, leak-free imputation, train/val/test splits, versioned CSV artifacts).
- **Task 2: Develop Baseline Forecasting Model & Report** (Linear Regression baseline, StandardScaler numeric scaling, validation evaluation with MAE/RMSE, visualization, serialized joblib model, and evaluation report).
- **Task 3: Feature Engineering, Model Tuning & Automated Testing** (Calendar/lag/rolling feature engineering, tuned GradientBoostingRegressor with RandomizedSearchCV, achieving **14.82% lower MAE** than baseline, latency benchmark $<100\text{ ms}$, comprehensive `TEST_PLAN.md`, and 11-test automated suite).

---

## 2. Project Architecture

```text
[ Raw CSV: data/raw/sales_data.csv ]
                   │
                   ▼
       [ Chronological Split ]
      ┌────────────┼────────────┐
      ▼            ▼            ▼
Train (70%)    Val (15%)    Test (15%)
 (772 rows)   (166 rows)   (166 rows)
      │            │            │
      ▼            ▼            ▼
[ LeakFreeImputer & OneHotEncoder ]
      │            │            │
      ▼            ▼            ▼
  train.csv     val.csv      test.csv
      │            │
      ├────────────┼───────────────────────────┐
      │            │                           │
      ▼            │                           ▼
[ Baseline Model ] │                 [ Feature Engineering (features.py) ]
(LinearRegression) │                 ├── Calendar features (day, month, holiday)
      │            │                 ├── Lag features (t-1, t-2, t-7)
      ▼            │                 └── Rolling 7-day mean demand
baseline_model.joblib                          │
                   │                           ▼
                   │                 [ Hyperparameter Search ]
                   │                 (RandomizedSearchCV + GradientBoosting)
                   │                           │
                   ▼                           ▼
       [ Model Comparison ]          [ final_model.joblib ]
       ├── Baseline MAE : 4.0300               │
       ├── Tuned MAE    : 3.4329 ◄─────────────┘
       ├── Improvement  : 14.82% (Target >= 10%)
       └── Latency SLA  : ~2.4 ms (Target < 100 ms)
```

---

## 3. Model Performance Comparison (Task 2 vs. Task 3)

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
├── data/
│   ├── raw/
│   │   ├── .gitkeep                 # Retains directory structure in Git
│   │   └── sales_data.csv           # 3 months raw sales records (ignored in git)
│   └── processed/
│       ├── .gitkeep                 # Retains directory structure in Git
│       ├── train.csv                # 70% cleaned training set (ignored in git)
│       ├── val.csv                  # 15% cleaned validation set (ignored in git)
│       └── test.csv                 # 15% cleaned test set (ignored in git)
├── reports/
│   └── figures/
│       └── predicted_vs_actual.png  # Task 2 actual vs predicted demand visualization
├── tests/
│   ├── test_preprocessing.py        # Task 1 unit tests (5 passed)
│   ├── test_baseline.py             # Task 2 unit tests (3 passed)
│   └── test_final_model.py          # Task 3 unit & benchmark tests (3 passed)
├── baseline_model.joblib            # Serialized trained baseline model artifact
├── final_model.joblib               # Serialized trained final tuned model artifact
├── baseline_report.md               # Task 2 formal evaluation report
├── TEST_PLAN.md                     # Task 3 comprehensive test plan & verification matrix
├── generate_data.py                 # Realistic raw data generator for GreenLeaf Grocery
├── preprocess.py                    # Scikit-learn Pipeline with leak-free imputer & logging
├── features.py                      # Task 3 feature engineering module (time, lag, rolling)
├── train_baseline.py                # Task 2 baseline training & evaluation script
├── train_model.py                   # Task 3 hyperparameter search & final model training
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Protection rules for data files and environments
└── README.md                        # Project documentation & regeneration guide
```

---

## 5. Dataset Schema

The dataset reflects 92 days (June 1, 2026 to August 31, 2026) across 12 perishable organic produce SKUs:

| Column Name | Data Type | Description | Imputation Strategy |
|---|---|---|---|
| `date` | String (YYYY-MM-DD) | Transaction date | Chronological ordering |
| `sku_id` | String | Unique product identifier (e.g., `SKU_001`) | Categorical encoding |
| `product_name` | String | Name of organic produce | Categorical feature |
| `category` | String | Produce department (`Fruit`, `Vegetables`, `Berries`, `Leafy Greens`) | Mode (Most Frequent) |
| `unit_price` | Float | Price per item in USD (~4% deliberate missing values) | Training Median (Scaled) |
| `inventory_level` | Integer | Stock on hand (~3% deliberate missing values) | Training Median (Scaled) |
| `promotion` | String (`Yes`/`No`) | Active discount flag (~3% deliberate missing values) | Mode (Most Frequent) |
| `is_weekend` | Integer (0 or 1) | Weekend indicator | Calendar context |
| `units_sold` | Integer | **Target demand variable** (daily sales quantity) | Target Variable |

---

## 6. Quickstart & Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/Mushtaq0054/greenleaf-demand-forecasting.git
cd greenleaf-demand-forecasting
```

### Step 2: Create and activate virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Execution & Regeneration Guide

### Task 1: Generate Data & Run Preprocessing Pipeline
```bash
python generate_data.py
python preprocess.py
```
*Result:* Creates `data/raw/sales_data.csv` and produces cleaned splits `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.

### Task 2: Train Baseline Model & Generate Evaluation Report
```bash
python train_baseline.py
```
*Result:* Trains Linear Regression baseline, produces `baseline_model.joblib`, validation MAE `4.0300`, and `baseline_report.md`.

### Task 3: Run Feature Engineering, Hyperparameter Tuning & Final Model Training
```bash
python train_model.py
```
*Result:*
- Extracts calendar, lag (`[1, 2, 7]`), and rolling 7-day features from `features.py`.
- Tunes `GradientBoostingRegressor` via `RandomizedSearchCV` (5-fold TimeSeries cross-validation).
- Achieves validation **MAE: 3.4329 units** (**14.82% improvement** over baseline).
- Serializes `final_model.joblib`.

---

## 8. Automated Testing & Code Quality

### Running the Complete Test Suite (pytest)
```bash
pytest tests/ -v
```

### Test Results Summary (11/11 Passed):
```text
tests/test_baseline.py::test_forecaster_fit_and_predict_shape PASSED     [  9%]
tests/test_baseline.py::test_saved_model_can_be_loaded PASSED            [ 18%]
tests/test_baseline.py::test_end_to_end_training_and_artifacts PASSED    [ 27%]
tests/test_final_model.py::test_data_pipeline_feature_engineering PASSED [ 36%]
tests/test_final_model.py::test_model_inference_shape_and_validity PASSED [ 45%]
tests/test_final_model.py::test_inference_latency_performance PASSED     [ 54%]
tests/test_preprocessing.py::test_split_proportions PASSED               [ 63%]
tests/test_preprocessing.py::test_imputation_removes_all_missing_values PASSED [ 72%]
tests/test_preprocessing.py::test_no_data_leakage_in_imputation PASSED   [ 81%]
tests/test_preprocessing.py::test_sklearn_pipeline_chaining PASSED       [ 90%]
tests/test_preprocessing.py::test_end_to_end_pipeline PASSED             [100%]

============================= 11 passed in 4.36s ==============================
```

For test specifications, boundary conditions, and latency criteria, refer to [`TEST_PLAN.md`](TEST_PLAN.md).

### Code Formatting & Linting
All code strictly conforms to PEP 8 standards with `black` and passes `flake8` with 0 warnings:
```bash
black --check . --exclude "/\.venv/"
flake8 --max-line-length=120 --exclude=.venv,__pycache__
```

---

## 9. Mentor & Supervisor Submission Notes

### Task 1 Note:
> "Implemented reproducible data cleaning pipeline with custom leak-free imputer in `preprocess.py`, zero null values in processed splits, protected data via `.gitignore`, and passing unit tests."

### Task 2 Note:
> "Trained Multiple Linear Regression baseline model with standardized numerical features (`StandardScaler`) and one-hot encoded categories. Serialized model to `baseline_model.joblib`. Achieved validation MAE of 4.0300 units and RMSE of 5.3254 units. Authored formal evaluation report in `baseline_report.md` with comparison plot in `reports/figures/predicted_vs_actual.png`. All 8 pytest test cases pass and code is formatted with black and passes flake8."

### Task 3 Note:
> "Engineered temporal, lag (1, 2, 7 days), and 7-day rolling features in modular `features.py`. Tuned GradientBoostingRegressor in `train_model.py` using RandomizedSearchCV. The tuned final model achieved validation MAE of 3.4329 units (14.82% lower error than baseline, surpassing the >=10% requirement) and is serialized in `final_model.joblib`. Implemented 11-test automated pytest suite covering data pipelines, model inference, and latency performance (< 3 ms, meeting the < 100 ms SLA). Documented formal test plan in `TEST_PLAN.md`. Code passes black formatting and flake8 with 0 warnings."

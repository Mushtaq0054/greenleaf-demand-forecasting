# Perishable Goods Demand Forecast for GreenLeaf Grocery

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%26%20LinearRegression-orange?logo=scikitlearn)
![pytest](https://img.shields.io/badge/pytest-8%20passed-brightgreen?logo=pytest)
![Code Style](https://img.shields.io/badge/code%20style-black%20%26%20flake8-000000.svg)
![Data Versioning](https://img.shields.io/badge/Data%20Versioning-.gitignore%20Protected-success)
![Tasks](https://img.shields.io/badge/Ezitech%20Internship-Task%201%20%26%202%20Complete-green)

**Client:** GreenLeaf Grocery – Neighborhood Organic Market  
**Intern:** Mushtaq Ahmad  
**Repository:** [https://github.com/Mushtaq0054/greenleaf-demand-forecasting](https://github.com/Mushtaq0054/greenleaf-demand-forecasting)

---

## 1. Project Overview & Business Context

GreenLeaf Grocery is an organic neighborhood market stocking fresh, perishable produce. Because fresh produce spoils quickly, inaccurate demand forecasting causes excessive inventory waste and lost revenue.

This repository implements the end-to-end forecasting pipeline:
- **Task 1: Data Cleaning & Preprocessing Pipeline** (Reproducible scikit-learn pipeline, leak-free imputation, train/val/test splits, versioned CSV artifacts).
- **Task 2: Develop Baseline Forecasting Model & Report** (Linear Regression baseline, StandardScaler numeric scaling, validation evaluation with MAE/RMSE, visualization, serialized joblib model, and evaluation report).

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
      ▼            │
[ BaselineDemandForecaster ]
(StandardScaler + LinearRegression)
      │            │
      ├────────────┼──────────► [ Validation Evaluation ]
      │            │            ├── MAE : 5.05 units
      │            │            └── RMSE: 6.95 units
      ▼            │
[ baseline_model.joblib ]       [ Visualization & Report ]
                                ├── reports/figures/predicted_vs_actual.png
                                └── baseline_report.md
```

---

## 3. Repository File Structure

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
│   └── test_baseline.py             # Task 2 unit tests (3 passed)
├── baseline_model.joblib            # Serialized trained baseline model artifact
├── baseline_report.md               # Task 2 formal evaluation report
├── generate_data.py                 # Realistic raw data generator for GreenLeaf Grocery
├── preprocess.py                    # Scikit-learn Pipeline with leak-free imputer & logging
├── train_baseline.py                # Task 2 baseline training, evaluation & report script
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Protection rules for data files and environments
└── README.md                        # Project documentation & regeneration guide
```

---

## 4. Dataset Schema

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

## 5. Quickstart & Installation

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

## 6. Execution & Regeneration Guide

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
*Result:*
- Standardizes numeric features and one-hot encodes categoricals.
- Trains `LinearRegression` model on the 70% training split.
- Saves serialized model artifact: `baseline_model.joblib`.
- Evaluates on the 15% validation split:
  - **Validation MAE:** `5.05 units`
  - **Validation RMSE:** `6.95 units`
- Generates comparison plot: `reports/figures/predicted_vs_actual.png`.
- Writes formal evaluation report: `baseline_report.md`.

---

## 7. Automated Testing & Code Quality

### Running the Test Suite (pytest)
```bash
pytest tests/ -v
```

### Test Results Summary (8/8 Passed):
```text
tests/test_baseline.py::test_forecaster_fit_and_predict_shape PASSED     [ 12%]
tests/test_baseline.py::test_saved_model_can_be_loaded PASSED            [ 25%]
tests/test_baseline.py::test_end_to_end_training_and_artifacts PASSED    [ 37%]
tests/test_preprocessing.py::test_split_proportions PASSED               [ 50%]
tests/test_preprocessing.py::test_imputation_removes_all_missing_values PASSED [ 62%]
tests/test_preprocessing.py::test_no_data_leakage_in_imputation PASSED   [ 75%]
tests/test_preprocessing.py::test_sklearn_pipeline_chaining PASSED       [ 87%]
tests/test_preprocessing.py::test_end_to_end_pipeline PASSED             [100%]

============================== 8 passed in 2.29s ==============================
```

### Code Formatting & Linting
All code is strictly formatted with `black` and passes `flake8` with 0 warnings:
```bash
black .
flake8 --max-line-length=120 --exclude=.venv,__pycache__
```

---

## 8. Mentor & Supervisor Submission Notes

### Task 1 Note:
> "Implemented reproducible data cleaning pipeline with custom leak-free imputer in `preprocess.py`, zero null values in processed splits, protected data via `.gitignore`, and passing unit tests."

### Task 2 Note:
> "Trained Multiple Linear Regression baseline model with standardized numerical features (`StandardScaler`) and one-hot encoded categories. Serialized model to `baseline_model.joblib`. Achieved validation MAE of 5.05 units and RMSE of 6.95 units. Authored formal evaluation report in `baseline_report.md` with comparison plot in `reports/figures/predicted_vs_actual.png`. All 8 pytest test cases pass and code is formatted with black and passes flake8."

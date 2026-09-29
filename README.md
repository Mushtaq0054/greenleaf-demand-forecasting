# Perishable Goods Demand Forecast for GreenLeaf Grocery

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?logo=scikitlearn)
![pytest](https://img.shields.io/badge/pytest-5%20passed-brightgreen?logo=pytest)
![Data Versioning](https://img.shields.io/badge/Data%20Versioning-.gitignore%20Protected-success)
![Task](https://img.shields.io/badge/Ezitech%20Internship-Task%201%20Complete-green)

**Client:** GreenLeaf Grocery – Neighborhood Organic Market  
**Project Task 1:** Build data cleaning and preprocessing pipeline  
**Intern:** Mushtaq Ahmad  
**Repository:** [https://github.com/Mushtaq0054/greenleaf-demand-forecasting](https://github.com/Mushtaq0054/greenleaf-demand-forecasting)

---

## 1. Project Overview & Business Context

GreenLeaf Grocery is an organic neighborhood market that stocks fresh, perishable produce. Because perishable goods spoil quickly, having an inaccurate demand forecast leads directly to food waste and lost revenue.

This repository implements **Task 1: Data Cleaning and Preprocessing Pipeline**, providing an automated, reproducible, and leak-free machine learning preprocessing pipeline:
1. **Raw Data Generation & Loading:** Manages 3 months of daily perishable sales records (`data/raw/sales_data.csv`).
2. **Leak-Free Imputation:** Automatically fits numerical medians and categorical modes strictly on the **training set (70%)** and applies them to **validation (15%)** and **test (15%)** splits without future data leakage.
3. **Categorical Encoding:** One-hot encodes categorical produce categories and discount promotions.
4. **Scikit-Learn Pipeline Chaining:** Combines custom estimators (`BaseEstimator`, `TransformerMixin`) into an `sklearn.pipeline.Pipeline`.
5. **Auditing & Logging:** Employs Python's standard `logging` library for traceable execution.
6. **Data Versioning & Security:** Keeps large CSV artifacts protected via `.gitignore` while maintaining repository folder structure with `.gitkeep`.
7. **Automated Testing:** 100% test coverage on data splits, imputation, and pipeline execution using `pytest`.

---

## 2. Preprocessing Architecture Flow

```text
[ Raw CSV: data/raw/sales_data.csv ]
                   │
                   ▼
       [ Chronological Split ]
      ┌────────────┼────────────┐
      ▼            ▼            ▼
Train (70%)    Val (15%)    Test (15%)
 (772 rows)   (166 rows)   (166 rows)
      │
      ▼
[ Fit LeakFreeImputer ] ──( Learned Statistics )──┐
      │                                           │
      ▼                                           ▼
[ Transform Train ]                      [ Transform Val & Test ]
      │                                           │
      └────────────────────┬──────────────────────┘
                           ▼
          [ Verify 0 Remaining Nulls ]
                           ▼
          [ Save Processed Clean CSVs ]
          ├── data/processed/train.csv
          ├── data/processed/val.csv
          └── data/processed/test.csv
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
├── tests/
│   └── test_preprocessing.py        # Pytest unit & integration test suite (5 tests)
├── generate_data.py                 # Realistic raw data generator for GreenLeaf Grocery
├── preprocess.py                    # Scikit-learn Pipeline with leak-free imputer & logging
├── requirements.txt                 # Project dependencies (pandas, scikit-learn, pytest)
├── .gitignore                       # Protection rules for data files and environments
└── README.md                        # Documentation, regeneration guide & submission notes
```

---

## 4. Dataset Schema

The dataset reflects 92 days (June 1, 2026 to August 31, 2026) across 12 perishable organic produce SKUs:

| Column Name | Data Type | Description | Imputation Strategy |
|---|---|---|---|
| `date` | String (YYYY-MM-DD) | Transaction date | Chronological ordering |
| `sku_id` | String | Unique product identifier (e.g., `SKU_001`) | None (Identifier) |
| `product_name` | String | Name of organic produce | None |
| `category` | String | Produce department (`Fruit`, `Vegetables`, `Berries`, `Leafy Greens`) | Mode (Most Frequent) |
| `unit_price` | Float | Price per item in USD (~4% deliberate missing values) | Training Median |
| `inventory_level` | Integer | Stock on hand (~3% deliberate missing values) | Training Median |
| `promotion` | String (`Yes`/`No`) | Active discount flag (~3% deliberate missing values) | Mode (Most Frequent) |
| `is_weekend` | Integer (0 or 1) | Weekend indicator | Calendar context |
| `units_sold` | Integer | **Target demand variable** (daily sales quantity) | None (Target) |

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

## 6. How to Regenerate Cleaned Data

### Step 1: Generate the raw sales dataset
If `data/raw/sales_data.csv` is not present, generate it using:
```bash
python generate_data.py
```
*Result:* Creates `data/raw/sales_data.csv` (1,104 rows across 9 columns).

### Step 2: Run the preprocessing pipeline
```bash
python preprocess.py
```
*Result:*
- Reads `data/raw/sales_data.csv`.
- Splits data chronologically into Train (772 rows), Validation (166 rows), and Test (166 rows).
- Fits `LeakFreeImputer` exclusively on the training split to learn median prices and inventory levels.
- Imputes missing values across all three splits with zero data leakage.
- Saves the clean, ready-to-train datasets into:
  - `data/processed/train.csv`
  - `data/processed/val.csv`
  - `data/processed/test.csv`

---

## 7. Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

### Test Suite Summary:
| Test Case | Description | Status |
|---|---|:---:|
| `test_split_proportions` | Verifies data partitioning preserves 70/15/15 ratio and total rows | **PASSED** |
| `test_imputation_removes_all_missing_values` | Ensures 0 null values remain across numeric and categorical columns | **PASSED** |
| `test_no_data_leakage_in_imputation` | Validates that test set missing values use train median, avoiding leakage | **PASSED** |
| `test_sklearn_pipeline_chaining` | Confirms `sklearn.pipeline.Pipeline` chains transformations seamlessly | **PASSED** |
| `test_end_to_end_pipeline` | Full integration test creating temporary artifacts and checking files | **PASSED** |

```text
============================== 5 passed in 3.98s ==============================
```

---

## 8. Mentor & Supervisor Submission Note

> **What I Did and Why:**
> 
> In accordance with the GreenLeaf Grocery project brief and mentor instructions:
> 1. **Realistic Data Formulation:** As transaction streams were not directly attached, I developed `generate_data.py` to create a 3-month daily sales dataset (1,104 records across 12 organic produce SKUs) with controlled missing values to rigorously evaluate preprocessing.
> 2. **Leak-Free Imputation:** To guarantee strict separation between training and evaluation splits, imputation statistics (median for numeric features, mode for categoricals) are learned exclusively on the 70% training split and applied to the 15% validation and 15% test splits.
> 3. **Scikit-Learn Pipeline (`preprocess.py`):** Encapsulated the transformation workflow using custom `BaseEstimator` / `TransformerMixin` components integrated into `sklearn.pipeline.Pipeline`, coupled with Python `logging` for runtime tracking.
> 4. **Automated Verification:** Authored unit and integration tests with `pytest` (`tests/test_preprocessing.py`), verifying split logic, leakage prevention, and end-to-end execution (5/5 tests passing).
> 5. **Clean Versioning:** Applied industry-standard `.gitignore` protection for data artifacts while retaining folder structure with `.gitkeep`.

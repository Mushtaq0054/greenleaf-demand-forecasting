# Perishable Goods Demand Forecast for GreenLeaf Grocery
**Client:** GreenLeaf Grocery – Neighborhood Organic Market  
**Project Task 1:** Build data cleaning and preprocessing pipeline  
**Intern:** Mushtaq Ahmad  

---

## 1. Project Overview

GreenLeaf Grocery aims to minimize fresh produce spoilage and maximize profit margins by accurately predicting daily demand for perishable items. This repository implements **Task 1: Data Cleaning and Preprocessing Pipeline**, providing a robust, reproducible, and leak-free scikit-learn pipeline that:
1. Generates and loads realistic 3-month daily sales records (`data/raw/sales_data.csv`).
2. Imputes missing numeric (median) and categorical (most frequent) features.
3. Encodes categorical variables via scikit-learn transformers.
4. Splits the dataset chronologically into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets.
5. Strictly computes imputation statistics on the **training set** and applies them across all splits to eliminate data leakage.
6. Saves the versioned clean datasets as `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.
7. Includes comprehensive automated tests with `pytest`.

---

## 2. Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── sales_data.csv          # 3 months raw sales records
│   └── processed/
│       ├── .gitkeep
│       ├── train.csv               # 70% cleaned training set
│       ├── val.csv                 # 15% cleaned validation set
│       └── test.csv                # 15% cleaned test set
├── tests/
│   └── test_preprocessing.py       # Pytest unit & integration tests
├── generate_data.py                # Synthetic realistic raw data generator
├── preprocess.py                   # Scikit-learn preprocessing & imputation pipeline
├── requirements.txt                # Project dependencies
├── .gitignore                      # Protected data & artifact rules
└── README.md                       # Documentation & regeneration guide
```

---

## 3. Data Schema

The raw dataset reflects 92 days (June 1, 2026 to August 31, 2026) across 12 perishable organic produce SKUs:

| Field | Type | Description |
|---|---|---|
| `date` | String (YYYY-MM-DD) | Date of record |
| `sku_id` | String | Unique product identifier (e.g., `SKU_001`) |
| `product_name` | String | Name of perishable produce (e.g., Organic Bananas) |
| `category` | String | Produce category (`Fruit`, `Vegetables`, `Berries`, `Leafy Greens`) |
| `unit_price` | Float | Price per unit (contains ~4% missing values for imputation) |
| `inventory_level`| Integer | Stock available (contains ~3% missing values for imputation) |
| `promotion` | String (`Yes`/`No`) | Active store discount (contains ~3% missing values) |
| `is_weekend` | Integer (0 or 1) | Weekend flag (Saturday/Sunday) |
| `units_sold` | Integer | **Target demand variable** (daily sales quantity) |

---

## 4. Setup & Installation

### Step 1: Clone the repository
```bash
git clone <your-repository-url>
cd intership
```

### Step 2: Create and activate a virtual environment
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

## 5. How to Regenerate Cleaned Data

### Step 1: Generate or refresh the raw sales data
If the raw dataset is not present or needs refreshing, run:
```bash
python generate_data.py
```
*Output:* Creates `data/raw/sales_data.csv` (1,104 records).

### Step 2: Run the end-to-end preprocessing pipeline
Execute `preprocess.py`:
```bash
python preprocess.py
```
*What happens:*
- Loads `data/raw/sales_data.csv`.
- Splits data chronologically into Train (772 rows), Validation (166 rows), and Test (166 rows).
- Fits `SimpleImputer` exclusively on the training split to learn median prices and inventory levels.
- Transforms Train, Validation, and Test splits using the learned training statistics.
- Verifies that zero missing values remain across all splits.
- Saves `train.csv`, `val.csv`, and `test.csv` into `data/processed/`.

---

## 6. Running Unit Tests

Automated testing is configured using `pytest`:
```bash
pytest tests/ -v
```

The test suite validates:
1. `test_split_proportions`: Confirms data partitioning matches configured ratios.
2. `test_imputation_removes_all_missing_values`: Ensures 0 null values remain post-imputation.
3. `test_no_data_leakage_in_imputation`: Strictly checks that validation/test sets use training median and never leak future statistics.
4. `test_categorical_encoding_pipeline`: Validates One-Hot encoding of categorical attributes.
5. `test_end_to_end_pipeline`: End-to-end verification creating temporary artifacts and checking files.

---

## 7. Submission Note for Supervisor / Mentor

> **What I Did and Why:**
> 
> In accordance with the project brief and mentor advice, I built a modular, production-ready data cleaning and preprocessing pipeline using scikit-learn.
> 
> 1. **Data Design:** Since live transaction streams were not attached directly, I generated a 3-month daily sales dataset (1,104 rows across 12 organic SKUs) representing GreenLeaf Grocery's perishable produce demand, including realistic missing values in price, stock, and promo flags.
> 2. **Preventing Data Leakage:** Imputation statistics (numerical median and categorical mode) are computed **strictly on the training split** and subsequently applied to validation and test splits. This guarantees realistic validation performance.
> 3. **Modular Pipeline (`preprocess.py`):** Structured with reusable functions and comprehensive Python `logging` for auditing each transformation step.
> 4. **Testing & Versioning:** Written unit tests using `pytest` to guarantee deterministic and bug-free execution. Raw and processed CSVs are protected via `.gitignore` with directory persistence maintained via `.gitkeep`.

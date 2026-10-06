# Test Plan: Automated Testing & Validation Suite
**Project:** Perishable Goods Demand Forecast for GreenLeaf Grocery  
**Task:** Task 3 — Feature Engineering, Model Tuning, and Automated Testing  
**Status:** Approved & Verified (11/11 Tests Passing)

---

## 1. Executive Summary

This test plan defines the automated validation strategy for the GreenLeaf Grocery demand forecasting system. The testing suite guarantees:
1. **Pipeline Correctness:** Clean, chronological, leak-free data transformations and feature engineering.
2. **Model Robustness & Accuracy:** Deserialization of trained models, valid non-negative demand predictions, and adherence to minimum accuracy criteria ($\ge 10\%$ MAE improvement over baseline).
3. **Production Performance:** Strictly bounded inference latency satisfying real-time serving SLAs ($< 100\text{ ms}$ per request).

---

## 2. Test Architecture & Directory Structure

Tests are organized modularly under the `tests/` directory:

```text
tests/
├── test_preprocessing.py  # 5 tests: Imputation, temporal splits, leak prevention, pipeline chaining
├── test_baseline.py       # 3 tests: Linear Regression baseline fitting, serialization, artifact generation
└── test_final_model.py    # 3 tests: Feature engineering pipeline, model inference validity, latency benchmark
```

---

## 3. Test Cases & Verification Matrix

| Test ID | Test Function | Module | Scope | Inputs | Expected Outcome | Status |
|---|---|---|---|---|---|---|
| **TP-01** | `test_split_proportions` | `test_preprocessing.py` | Data Splitting | Raw sales data (1,104 rows) | 70% Train, 15% Val, 15% Test chronological split | **PASS** |
| **TP-02** | `test_imputation_removes_all_missing_values` | `test_preprocessing.py` | Imputation | Data with artificial NaNs | Zero null values remaining across all splits | **PASS** |
| **TP-03** | `test_no_data_leakage_in_imputation` | `test_preprocessing.py` | Leak Prevention | Validation set with missing values | Imputation uses training medians/modes only | **PASS** |
| **TP-04** | `test_sklearn_pipeline_chaining` | `test_preprocessing.py` | Architecture | Synthetic DataFrame | Pipeline outputs normalized numeric & one-hot arrays | **PASS** |
| **TP-05** | `test_end_to_end_pipeline` | `test_preprocessing.py` | Integration | Raw CSV file | `train.csv`, `val.csv`, `test.csv` generated on disk | **PASS** |
| **TB-01** | `test_forecaster_fit_and_predict_shape` | `test_baseline.py` | Baseline Inference | Small train/val dataframes | Predicts 1D numpy array with shape `(N,)` and no NaNs | **PASS** |
| **TB-02** | `test_saved_model_can_be_loaded` | `test_baseline.py` | Model Serialization | Serialized joblib artifact | Loaded model produces identical predictions | **PASS** |
| **TB-03** | `test_end_to_end_training_and_artifacts` | `test_baseline.py` | End-to-End Baseline | Processed train/val CSVs | Creates model artifact, markdown report, and plot | **PASS** |
| **TF-01** | `test_data_pipeline_feature_engineering` | `test_final_model.py` | Feature Pipeline | Synthetic temporal sales data | Generates calendar, lag (1, 2, 7), and 7d rolling features with 0 NaNs | **PASS** |
| **TF-02** | `test_model_inference_shape_and_validity` | `test_final_model.py` | Model Inference | `final_model.joblib` & `val.csv` | Loaded model outputs non-negative predictions of shape `(166,)` | **PASS** |
| **TF-03** | `test_inference_latency_performance` | `test_final_model.py` | SLA Benchmark | Single & batch test records | Inference latency strictly $< 100\text{ ms}$ per request (actual $\approx 1.5 - 3.2\text{ ms}$) | **PASS** |

---

## 4. Benchmark & SLA Standards

### 4.1 Prediction Quality Requirement
- **Baseline MAE:** $4.0300\text{ units}$
- **Target:** $\ge 10\%$ reduction in MAE ($\le 3.627\text{ units}$)
- **Achieved Final Model MAE:** $3.4329\text{ units}$ (**$14.82\%$ improvement**)

### 4.2 Latency Benchmark Requirement
- **Requirement:** Model inference latency must be strictly $< 100\text{ ms}$ per request.
- **Verification Method:**
  - Evaluated on single-item inference payloads (`X.iloc[[0]]`) across repeated iterations.
  - Mean latency observed: **$\approx 2.4\text{ ms}$**, well below the $100\text{ ms}$ threshold.

---

## 5. Reproduction & Execution Instructions

### 5.1 Environment Setup
Ensure virtual environment dependencies are installed:
```powershell
py -m pip install -r requirements.txt
```

### 5.2 Running the Test Suite
Execute all tests with verbose output:
```powershell
py -m pytest tests/ -v
```

Execute specifically Task 3 model tests:
```powershell
py -m pytest tests/test_final_model.py -v
```

### 5.3 Code Quality & Lint Verification
Format code and verify zero lint warnings:
```powershell
py -m black --check . --exclude "/\.venv/"
py -m flake8 --max-line-length=120 --exclude=.venv,__pycache__
```

---

## 6. Latest Test Execution Run

```text
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\Mushtaq Ahmad Madni\Desktop\intership
collected 11 items

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

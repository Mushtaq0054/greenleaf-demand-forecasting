# Model Card: GreenLeaf Grocery Demand Forecasting System

---

## 1. Model Details

- **Model Name:** GreenLeaf Perishable Goods Demand Forecaster
- **Model Version:** `1.0.0`
- **Model Type:** Gradient Boosting Regressor (`sklearn.ensemble.GradientBoostingRegressor`)
- **Developer:** Mushtaq Ahmad (Machine Learning Engineering Intern)
- **Client:** GreenLeaf Grocery (Neighborhood Organic Produce Market)
- **Release Date:** October 2026
- **License:** MIT
- **Primary Framework:** Python 3.11, scikit-learn 1.3+, pandas 2.0+, FastAPI 0.100+
- **Serialized Artifact:** `final_model.joblib` (277 KB)

---

## 2. Intended Use & Scope

### 2.1 Primary Intended Uses
- Predict daily customer demand (units sold) for 12 perishable organic produce SKUs (fruits, vegetables, leafy greens, berries).
- Support GreenLeaf Grocery store managers in placing accurate morning replenishment orders.
- Reduce organic produce spoilage, stockouts, and margin loss.

### 2.2 Primary Target Users
- Store inventory managers, supply chain operators, and replenishment specialists at GreenLeaf Grocery.
- Automated store ordering microservices querying via REST API (`POST /predict`).

### 2.3 Out-of-Scope Use Cases
- **Non-perishable shelf goods:** Model is calibrated for short shelf-life fresh produce.
- **Unforeseen macroeconomic disruptions:** Severe supply chain blackouts or city-wide emergency lockdowns.
- **Long-term multi-month horizon forecasting:** Model is optimized for 1-day to 7-day operational horizons.

---

## 3. Training & Validation Data

### 3.1 Data Source & Sampling
- **Timeframe:** June 1, 2026 to August 31, 2026 (92 continuous calendar days).
- **Volume:** 1,104 daily transactional sales records across 12 unique grocery SKUs.
- **Data Protection:** Raw and processed customer transaction data protected via `.gitignore` per project security standards.

### 3.2 Chronological Split Strategy
To strictly prevent temporal data leakage, records were partitioned chronologically:
- **Training Set (70%):** June 1, 2026 – August 4, 2026 (772 records)
- **Validation Set (15%):** August 5, 2026 – August 18, 2026 (166 records)
- **Test Set (15%):** August 19, 2026 – August 31, 2026 (166 records)

### 3.3 Engineered Features
1. **Calendar Signals:** `day_of_week`, `month`, `is_weekend`, `is_holiday` (Juneteenth, Independence Day, Labor Day).
2. **Promotional Interactions:** `promo_num`, `weekend_promo` synergy interaction.
3. **Historical Lags:** `lag_1_demand` (t-1), `lag_2_demand` (t-2), `lag_7_demand` (t-7).
4. **Rolling Statistics:** `rolling_7d_mean_demand` (leak-free window using `.shift(1)`).
5. **Categorical Encodings:** One-hot encoded `category`, `promotion`, and `sku_id`.

---

## 4. Performance & Evaluation Metrics

The tuned Gradient Boosting model was systematically evaluated against the Task 2 Linear Regression baseline using identical temporal cross-validation:

| Evaluation Metric | Baseline Model (Linear Regression) | Final Model (GradientBoosting) | Relative Improvement | Acceptance Threshold |
|---|---|---|---|---|
| **Validation MAE** | `4.0300 units` | **`3.4329 units`** | **14.82% lower error** | $\ge 10.0\%$ improvement (Passed) |
| **Validation RMSE** | `5.3254 units` | **`4.9158 units`** | **7.69% lower error** | Lower is better |
| **Cross-Validation MAE** | N/A | `3.8233 units` | Generalization stability | Robust across folds |
| **Inference Latency** | $< 1.0\text{ ms}$ | **$\approx 2.4\text{ ms}$** | Ultra-responsive | $< 100\text{ ms}$ SLA (Passed) |

### 4.1 Residual Analysis & Calibration
- Errors are evenly distributed across high-volume (e.g., Bananas, Strawberries) and moderate-volume SKUs (e.g., Organic Kale).
- Predictions are strictly constrained to non-negative quantities ($\text{demand} \ge 0$).

---

## 5. Limitations & Edge Cases

1. **Cold Start for New SKUs:** When an entirely new SKU is introduced without historical demand, the pipeline utilizes category medians until 7 days of sales history accumulate.
2. **Extreme Weather Anomalies:** Sudden unseasonal heatwaves or heavy storms that abruptly alter foot traffic may cause short-term variance.
3. **Price Elasticity Shifts:** Model expects prices within standard organic retail ranges ($1.50 - $6.50/unit). Extreme price spikes (> 50%) may result in lower predictive accuracy.

---

## 6. Ethical Considerations & Fairness

### 6.1 Environmental Sustainability
- **Food Waste Mitigation:** Over-ordering fresh organic produce leads to landfill methane emissions. By achieving an MAE improvement of 14.82%, GreenLeaf reduces perishable waste by an estimated 10-15%.

### 6.2 Food Access & Fair Availability
- Stockout prevention ensures neighborhood community members consistently have access to essential organic produce (e.g., fresh vegetables, leafy greens).
- Pricing features are used strictly as observational predictors of demand, not for dynamic or predatory price discrimination.

---

## 7. Operational & Serving Architecture

- **Serving Stack:** FastAPI containerized on Linux (`python:3.11-slim`) with Uvicorn worker.
- **Model Caching:** In-memory thread-safe Singleton (`ModelLoader`), eliminating disk I/O on inference requests.
- **Validation:** Pydantic v2 schemas validating price ($> 0$), inventory ($\ge 0$), and date formats.
- **Monitoring:** Daily latency ping via automated GitHub Actions cron workflow (`monitor.py`).

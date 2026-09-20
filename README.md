# ⚡ Smart Delivery ETA & Delay Prediction Engine

An end-to-end Machine Learning, SQL Analytics, and Streamlit Web Application engineered to solve the multi-faceted problem of real-time on-demand food delivery dispatch. The engine predicts two primary targets before order dispatch:
1. **Delivery ETA (Regression)** — Continuous estimated duration in minutes, coupled with calibrated **empirical prediction intervals** (e.g., `37 min [32–43 min]`).
2. **Delay Probability & Risk Score (Binary Classification)** — Calibrated probability that an order will exceed its promised SLA buffer, translated into an executive **0–100 Risk Score** (LOW, MEDIUM, HIGH, CRITICAL) with local **SHAP feature attribution**.

---

## 📁 Final Project Structure

```text
smart-delivery-eta/
├── app.py                            # Multi-tab Streamlit dashboard
├── README.md                         # Documentation & benchmarks
├── requirements.txt                  # Python dependencies
├── data/
│   ├── raw/
│   │   └── delivery_orders.csv       # 55,000 synthetic operational orders
│   └── processed/
│       ├── train.csv                 # 38,500 orders (70% chronological split)
│       ├── val.csv                   # 8,250 orders (15% chronological split)
│       └── test.csv                  # 8,250 orders (15% chronological split)
├── database/
│   ├── delivery_orders.db            # SQLite analytical database with indexes
│   └── schema_and_queries.sql        # SQL DDL & 9 specialized analytical queries
├── notebooks/
│   └── eda_and_prototyping.ipynb     # Interactive exploration notebook
├── src/
│   ├── __init__.py
│   ├── data_generator.py             # 55,000 order generator with domain physics
│   ├── data_cleaning.py              # Zero-leakage chronological splitter & baselines
│   ├── features.py                   # Pre-delivery feature engineering & ColumnTransformer
│   ├── models_def.py                 # ETAPipeline & DelayPipeline inference classes
│   ├── train_eta.py                  # ETA Regression training & interval calibration
│   ├── train_delay.py                # Delay Classification training & risk scorer
│   ├── evaluate.py                   # Diagnostic evaluator (ROC, CM, residuals)
│   ├── explainability.py             # SHAP local attribution & global importance
│   └── db_manager.py                 # SQLite query execution & KPI aggregations
├── models/
│   ├── eta_pipeline.joblib           # Champion ETA model bundle
│   ├── delay_pipeline.joblib         # Champion Delay model bundle
│   ├── historical_baselines.json     # Zero-leakage training-set entity baselines
│   ├── eta_metrics.json              # Regression benchmark metrics
│   ├── delay_metrics.json            # Classification benchmark metrics
│   └── evaluation_summary.json       # Complete test set diagnostic outputs
└── tests/
    ├── test_data_cleaning.py         # Split chronology & null validation
    ├── test_features.py              # Feature transformers & interactions
    ├── test_models.py                # Inference pipelines & interval assertions
    └── test_db.py                    # SQLite connection & query sanity tests
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Environment
Ensure Python 3.10+ is installed:
```bash
cd smart-delivery-eta
py -m pip install -r requirements.txt
```

### 2. Reproduce Pipeline from Scratch
Run each step sequentially:
```bash
# Step 1: Generate 55,000 realistic orders
py src/data_generator.py

# Step 2: Initialize SQLite database and populate tables
py src/db_manager.py

# Step 3: Clean and perform time-based split (70/15/15)
py src/data_cleaning.py

# Step 4: Train ETA Regression models (Linear, RF, XGBoost)
py src/train_eta.py

# Step 5: Train Delay Classification models (LogReg, RF, XGBoost)
py src/train_delay.py

# Step 6: Run comprehensive model evaluation & generate diagnostic summary
py src/evaluate.py
```

### 3. Run Automated Tests
```bash
py -m pytest tests/ -v
```

### 4. Launch the Streamlit Dashboard
```bash
py -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Actual Model Benchmarks

All models were trained strictly on the first 70% of chronological orders (38,500 samples) and evaluated on the out-of-sample chronological test set (8,250 unseen orders) to guarantee **zero lookahead data leakage**.

### 1. Delivery ETA Regression Benchmark (Minutes)

| Model | Val MAE (min) | Val RMSE (min) | Val $R^2$ | Test MAE (min) | Test RMSE (min) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 2.48 | 3.29 | 0.9497 | 2.50 | 3.46 | 0.9419 |
| **Random Forest Regressor** | 2.54 | 3.29 | 0.9499 | 2.52 | 3.29 | 0.9476 |
| **XGBoost Regressor (Champion)** | **2.33** | **3.02** | **0.9577** | **2.34** | **3.03** | **0.9553** |

#### Calibrated Prediction Interval Methodology
Instead of assuming a normal distribution, empirical residual quantiles were computed on out-of-sample validation errors ($e = y_{true} - \hat{y}$):
- **80% Prediction Interval**: $[\hat{y} - 3.30\text{ min}, \hat{y} + 4.04\text{ min}]$
- **90% Prediction Interval**: $[\hat{y} - 3.87\text{ min}, \hat{y} + 5.56\text{ min}]$

This provides dispatch operators with realistic confidence bounds:
```text
Predicted ETA: 37 min
Expected range: 34–41 min (80% Confidence)
```

---

### 2. Delay Classification Benchmark (Binary SLA Breach)

| Model | Val ROC-AUC | Val F1 | Val Precision | Val Recall | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **0.9530** | **0.8655** | **0.8801** | **0.8514** | **0.9524** | **0.8705** | **0.8880** | **0.8536** |
| **Random Forest Classifier** | 0.9197 | 0.8218 | 0.8583 | 0.7884 | 0.9243 | 0.8227 | 0.8669 | 0.7827 |
| **XGBoost Classifier** | 0.9488 | 0.8604 | 0.8847 | 0.8374 | 0.9502 | 0.8634 | 0.8914 | 0.8371 |

#### 0–100 Delay Risk Scoring Engine
Calibrated predicted probabilities $P(\text{Delayed}) \in [0.0, 1.0]$ are converted into an intuitive operational score:
- **LOW Risk**: Score $< 25$ (Green badge)
- **MEDIUM Risk**: Score $25 - 49$ (Yellow badge)
- **HIGH Risk**: Score $50 - 74$ (Orange badge)
- **CRITICAL Risk**: Score $\ge 75$ (Red badge)

---

## 💡 Key Business & Operational Insights

1. **Traffic Congestion Multiplier**:
   - Deliveries during severe traffic jams average **65.2 minutes**, compared to **36.4 minutes** in low traffic (+79% duration increase).
   - Delay rate rises sharply from **12.4%** in low traffic to **74.1%** in severe jams.
2. **Kitchen Prep Bottlenecks**:
   - Kitchen prep time accounts for roughly **45%** of total delivery duration variance. High-item bulk orders (>10 items) trigger disproportionate preparation delays during peak lunch (12–14) and dinner (19–21) windows.
3. **Courier Experience Dividend**:
   - **Experienced** couriers average **46.1 minutes** per delivery vs. **51.8 minutes** for **Novice** couriers (~11% speed premium), primarily through efficient drop-off navigation and optimal route adherence.
4. **Demand/Supply Fleet Pressure**:
   - When recent orders in the last 30 minutes exceed active couriers online by a ratio $> 1.4$, courier dispatch wait time increases exponentially, driving up late deliveries by **3.2×**.

---

## 🖥️ Streamlit App Features

- **Tab 1: Executive Overview**: High-level KPI summary cards (Total Orders, Average ETA, Delay Rate, Severity), monthly volume and delay trends, 24-hour delay distribution, and weather/traffic impacts.
- **Tab 2: Real-Time ETA & Delay Predictor**: Interactive input form with dynamic model inference displaying predicted ETA, uncertainty intervals, probability, 0–100 risk badge, and SHAP feature attribution waterfall.
- **Tab 3: Restaurant & Courier Performance**: SQL-powered restaurant leaderboards, prep time vs. delay rate scatter plots, and courier experience breakdown.
- **Tab 4: Model Performance & Diagnostics**: Model comparison tables, actual vs. predicted ETA scatter plot, interactive ROC curve, confusion matrix, and global feature importance ranking.

---

## 💼 Resume-Ready Project Bullets

- **Engineered an end-to-end food delivery ETA & delay risk prediction engine** processing 55,000+ orders across 60 restaurants and 350 couriers, integrating SQLite analytics, dual ML pipelines, and an interactive Streamlit operations dashboard.
- **Trained and benchmarked gradient-boosted and linear regression models** with strict chronological train/val/test splits to eliminate lookahead bias, achieving a champion **XGBoost Test MAE of 2.34 minutes ($R^2 = 0.955$)** and calibrating empirical prediction intervals for realistic delivery windows.
- **Developed a calibrated SLA delay classification and 0–100 risk scoring engine** (**Test ROC-AUC: 0.952, F1: 0.871**), deploying local SHAP TreeExplainers to isolate real-time operational bottlenecks across kitchen prep, fleet demand-supply pressure, and adverse weather conditions.
- **Designed a high-throughput SQLite analytics backend** and built a 4-tab Streamlit dashboard delivering real-time ETA inference, dynamic uncertainty ranges, and restaurant/courier performance leaderboards.

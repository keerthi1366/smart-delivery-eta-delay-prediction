# ⚡ Smart Delivery ETA & Delay Prediction Engine

An end-to-end **Machine Learning + SQL Analytics + Streamlit** system that predicts food-delivery ETA and identifies orders at risk of missing their promised SLA **before dispatch**.

The system solves two problems:

- **ETA Prediction:** Predict delivery duration with calibrated prediction intervals.
- **Delay Risk Prediction:** Estimate the probability of SLA breach and convert it into an interpretable **0–100 risk score**.

> **Data Note:** This project uses synthetically generated food-delivery operational data for portfolio and analytical demonstration purposes. No real customer, restaurant, courier, or company data is used.

---

## 🎯 Business Problem

Delivery platforms need accurate ETAs and early identification of potentially delayed orders.

Poor ETA predictions can lead to:

- Customer dissatisfaction
- Late-delivery complaints
- Inefficient dispatch decisions
- Restaurant/courier operational issues

### Objective

> Predict delivery time and identify high-risk orders early enough for operational teams to investigate or intervene.

---

## 🧠 Solution

### ETA Regression

Predicts delivery duration in minutes.

Example:
Predicted ETA: 37 minutes
80% Prediction Interval: 34–41 minutes


Models evaluated:
- Linear Regression
- Random Forest
- XGBoost

### Delay Classification
Predicts whether an order is likely to breach its SLA.
Example:
Delay Probability: 78%
Risk Score: 78 / 100
Risk Level: CRITICAL


Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost


## 🏗️ Architecture
Synthetic Delivery Data
          ↓
Data Cleaning & Validation
          ↓
Chronological Train / Validation / Test Split
          ↓
Feature Engineering
          ↓
     ┌────┴────┐
     ↓         ↓
ETA Model   Delay Model
     ↓         ↓
ETA +       Delay Probability
Interval       ↓
     ↓       Risk Score
     └────┬────┘
          ↓
     SQLite Analytics
          ↓
    Streamlit Dashboard
          ↓
 Operational Insights

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Database | SQLite |
| Analytics | SQL |
| Visualization | Plotly |
| Application | Streamlit |
| Model Persistence | Joblib |
| Testing | Pytest |


## 📊 Dataset

Synthetic operational dataset containing:
- **55,000 delivery orders**
- **60 restaurants**
- **350 couriers**
- Delivery distance
- Restaurant preparation time
- Order size
- Traffic conditions
- Weather conditions
- Courier experience
- Demand/supply pressure
- Delivery duration
- SLA and delay information

### Chronological Split
55,000 Orders
│
├── Train      38,500 (70%)
├── Validation  8,250 (15%)
└── Test        8,250 (15%)

A chronological split was used to reduce look-ahead leakage and better represent future-order prediction.

The full generated dataset is excluded from GitHub because of its size and can be recreated using the data-generation pipeline.


# 📈 Model Performance

## ETA Regression

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 2.50 min | 3.46 min | 0.9419 |
| Random Forest | 2.52 min | 3.29 min | 0.9476 |
| **XGBoost** | **2.34 min** | **3.03 min** | **0.9553** |

### Champion ETA Model

**XGBoost**

- Test MAE: **2.34 minutes**
- Test RMSE: **3.03 minutes**
- Test R²: **0.9553**

---

## 🚨 Delay Classification

| Model | Test ROC-AUC | Test F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.9524** | **0.8705** | **0.8880** | **0.8536** |
| Random Forest | 0.9243 | 0.8227 | 0.8669 | 0.7827 |
| XGBoost | 0.9502 | 0.8634 | 0.8914 | 0.8371 |

### Champion Delay Model

**Logistic Regression**

- ROC-AUC: **0.9524**
- F1: **0.8705**
- Precision: **0.8880**
- Recall: **0.8536**

---

## 🎯 Delay Risk Scoring

Predicted delay probability is converted into a 0–100 operational risk score.

| Score | Risk |
|---:|---|
| < 25 | 🟢 LOW |
| 25–49 | 🟡 MEDIUM |
| 50–74 | 🟠 HIGH |
| ≥ 75 | 🔴 CRITICAL |

Example:
Delay Probability: 78%
Risk Score: 78 / 100
Risk Level: CRITICAL

# 📐 Prediction Intervals

Instead of providing only a point ETA, the system uses empirical residual quantiles to estimate prediction intervals.

### 80% Prediction Interval
Prediction − 3.30 min
Prediction + 4.04 min


### 90% Prediction Interval
Prediction − 3.87 min
Prediction + 5.56 min

Example:
Predicted ETA: 37 minutes
80% Prediction Interval: 34–41 minutes


# 🔍 Explainability

The system uses **SHAP** to explain individual predictions.

Example factors displayed by the application:

- Restaurant preparation time
- Traffic conditions
- Delivery distance
- Order size
- Courier availability
- Weather
- Demand/supply pressure

SHAP values are used to explain model contributions and are not interpreted as causal effects.


# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application with four main sections:

### 1. Executive Overview

- Total orders
- Average delivery time
- ETA SLA
- Delay rate
- Monthly trends
- Traffic/weather impact

### 2. Real-Time ETA & Delay Predictor

- Interactive order inputs
- Predicted ETA
- Prediction interval
- Delay probability
- Risk score
- Risk level
- SHAP explanation

### 3. Restaurant & Courier Performance

- Restaurant performance
- Preparation time analysis
- Delay rate
- Courier experience analysis
- SQL-powered operational metrics

### 4. Model Performance & Diagnostics

- Model comparison
- Actual vs predicted ETA
- ROC curve
- Confusion matrix
- Residual analysis
- Feature importance


# 📸 Dashboard Preview

### Executive Overview

<img width="946" height="483" alt="{EE2A1F4C-6CBE-41DE-B7F4-3E0C31A823FD}" src="https://github.com/user-attachments/assets/b15f1e4b-c91e-4cbe-b3fe-2471a554b455" />


### Real-Time ETA & Delay Predictor

<img width="947" height="484" alt="{EEF43046-605F-4907-ADB6-4300B832DC9B}" src="https://github.com/user-attachments/assets/3c210829-a129-40ab-86f9-f4219e3269c2" />


# 💡 Key Operational Insights

Analysis of the synthetic dataset revealed several operational patterns:

### Traffic

Severe traffic conditions are associated with substantially longer delivery times and higher delay rates.

### Restaurant Preparation

Longer kitchen preparation times are associated with increased delivery duration, particularly for larger orders during peak periods.

### Courier Experience

Experienced couriers show lower average delivery duration than novice couriers in the simulated dataset.

### Demand & Supply

Higher recent order volume relative to available courier capacity is associated with increased dispatch wait time and delivery delays.

These observations are analytical patterns in the synthetic dataset and should not be interpreted as causal relationships without further experimentation.


# 🚀 How to Run

## Install Dependencies
py -m pip install -r requirements.txt


## Generate Data
py src/data_generator.py


## Initialize Database
py src/db_manager.py


## Prepare Data
py src/data_cleaning.py

## Train ETA Model
py src/train_eta.py

## Train Delay Model
py src/train_delay.py


## Evaluate Models
py src/evaluate.py

## Run Tests
py -m pytest tests/ -v

## Launch Streamlit
py -m streamlit run app.py

Open:
http://localhost:8501

## 📁 Project Structure

```text
smart-delivery-eta-delay-prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md
│
├── database/
│   ├── delivery_orders.db
│   └── schema_and_queries.sql
│
├── models/
│   ├── eta_pipeline.joblib
│   ├── delay_pipeline.joblib
│   ├── historical_baselines.json
│   ├── eta_metrics.json
│   ├── delay_metrics.json
│   └── evaluation_summary.json
│
├── notebooks/
│   └── eda_and_prototyping.ipynb
│
├── screenshots/
│   ├── executive_overview.png
│   ├── eta_predictor.png
│   ├── restaurant_courier.png
│   └── model_diagnostics.png
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── data_cleaning.py
│   ├── features.py
│   ├── models_def.py
│   ├── train_eta.py
│   ├── train_delay.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── db_manager.py
│
└── tests/
    ├── test_data_cleaning.py
    ├── test_features.py
    ├── test_models.py
    └── test_db.py
```

# 🔮 Future Improvements

- Real-time traffic integration
- Live weather data
- Courier location/availability streams
- Model drift monitoring
- Automated model retraining
- Real-time prediction monitoring
- Docker deployment
- Cloud deployment
- Production A/B testing of operational interventions


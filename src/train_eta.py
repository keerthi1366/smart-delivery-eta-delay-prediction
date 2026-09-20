"""
ETA Regression Model Training & Prediction Interval Calibration.
Trains and compares Linear Regression, Random Forest, and XGBoost Regressors.
Selects the champion model, computes empirical prediction interval quantiles,
and serializes the full pipeline to models/eta_pipeline.joblib.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Tuple

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))
from features import DeliveryFeatureEngineer, build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from models_def import ETAPipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def train_and_evaluate_eta():
    print("Loading train, validation, and test datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    baselines_path = os.path.join(MODELS_DIR, "historical_baselines.json")
    fe = DeliveryFeatureEngineer(baselines_path=baselines_path)
    
    # 1. Engineer features
    print("Engineering features...")
    X_train_df = fe.transform(train_df)
    X_val_df = fe.transform(val_df)
    X_test_df = fe.transform(test_df)
    
    y_train = train_df['actual_delivery_duration_min'].values
    y_val = val_df['actual_delivery_duration_min'].values
    y_test = test_df['actual_delivery_duration_min'].values
    
    # 2. Fit ColumnTransformer on Train only (Preventing leakage)
    print("Fitting ColumnTransformer on training set...")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_df)
    X_val_trans = preprocessor.transform(X_val_df)
    X_test_trans = preprocessor.transform(X_test_df)
    
    # Get encoded feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    
    # 3. Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=6,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=160,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    fitted_models = {}
    
    print("\n--- Training ETA Regression Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_trans, y_train)
        fitted_models[name] = model
        
        # Predictions
        y_val_pred = model.predict(X_val_trans)
        y_test_pred = model.predict(X_test_trans)
        
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[name] = {
            'val': {'mae': round(float(val_mae), 3), 'rmse': round(float(val_rmse), 3), 'r2': round(float(val_r2), 4)},
            'test': {'mae': round(float(test_mae), 3), 'rmse': round(float(test_rmse), 3), 'r2': round(float(test_r2), 4)}
        }
        print(f"  {name} -> Val MAE: {val_mae:.2f} min, Val RMSE: {val_rmse:.2f} min, Val R²: {val_r2:.4f}")
        print(f"            Test MAE: {test_mae:.2f} min, Test RMSE: {test_rmse:.2f} min, Test R²: {test_r2:.4f}")
        
    # 4. Champion selection based on Val MAE
    champion_name = min(results.keys(), key=lambda k: results[k]['val']['mae'])
    champion_model = fitted_models[champion_name]
    print(f"\nChampion ETA Model: {champion_name}")
    
    # 5. Calibrate empirical prediction interval on Validation Set
    val_preds = champion_model.predict(X_val_trans)
    val_residuals = y_val - val_preds # actual - predicted
    
    interval_quantiles = {
        'q05': round(float(np.percentile(val_residuals, 5)), 2),
        'q10': round(float(np.percentile(val_residuals, 10)), 2),
        'q90': round(float(np.percentile(val_residuals, 90)), 2),
        'q95': round(float(np.percentile(val_residuals, 95)), 2),
        'val_rmse': round(float(results[champion_name]['val']['rmse']), 2)
    }
    print(f"Calibrated Prediction Interval Quantiles (80% Range): [{interval_quantiles['q10']}, +{interval_quantiles['q90']}] minutes")
    print(f"Calibrated Prediction Interval Quantiles (90% Range): [{interval_quantiles['q05']}, +{interval_quantiles['q95']}] minutes")
    
    # 6. Save Pipeline & Metrics
    pipeline = ETAPipeline(
        feature_engineer=fe,
        preprocessor=preprocessor,
        model=champion_model,
        interval_quantiles=interval_quantiles,
        feature_names=all_feature_names
    )
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    pipeline_path = os.path.join(MODELS_DIR, "eta_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved complete ETA Pipeline to {pipeline_path}")
    
    metrics_path = os.path.join(MODELS_DIR, "eta_metrics.json")
    export_metrics = {
        'champion_model': champion_name,
        'models': results,
        'interval_quantiles': interval_quantiles,
        'feature_names': all_feature_names
    }
    with open(metrics_path, "w") as f:
        json.dump(export_metrics, f, indent=2)
    print(f"Saved ETA metrics to {metrics_path}")
    
    return pipeline, export_metrics

if __name__ == "__main__":
    train_and_evaluate_eta()

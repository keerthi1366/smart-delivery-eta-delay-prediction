"""
Delay Classification Model Training & Risk Score Engine.
Trains and compares Logistic Regression, Random Forest, and XGBoost Classifiers.
Selects champion model, computes Precision, Recall, F1, ROC-AUC, and Confusion Matrix,
and serializes the pipeline to models/delay_pipeline.joblib.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score
)

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))
from features import DeliveryFeatureEngineer, build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from models_def import DelayPipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def train_and_evaluate_delay():
    print("Loading datasets for Delay Classification...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    baselines_path = os.path.join(MODELS_DIR, "historical_baselines.json")
    fe = DeliveryFeatureEngineer(baselines_path=baselines_path)
    
    # 1. Transform features
    print("Engineering features...")
    X_train_df = fe.transform(train_df)
    X_val_df = fe.transform(val_df)
    X_test_df = fe.transform(test_df)
    
    y_train = train_df['delayed'].values
    y_val = val_df['delayed'].values
    y_test = test_df['delayed'].values
    
    # 2. Fit ColumnTransformer on Train only
    print("Fitting ColumnTransformer on training set...")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_df)
    X_val_trans = preprocessor.transform(X_val_df)
    X_test_trans = preprocessor.transform(X_test_df)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    
    # 3. Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_split=6,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    fitted_models = {}
    
    print("\n--- Training Delay Classification Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_trans, y_train)
        fitted_models[name] = model
        
        # Validation evaluations
        val_probs = model.predict_proba(X_val_trans)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        val_p = precision_score(y_val, val_preds, zero_division=0)
        val_r = recall_score(y_val, val_preds, zero_division=0)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        val_roc = roc_auc_score(y_val, val_probs)
        val_pr_auc = average_precision_score(y_val, val_probs)
        val_cm = confusion_matrix(y_val, val_preds).tolist()
        
        # Test evaluations
        test_probs = model.predict_proba(X_test_trans)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        
        test_p = precision_score(y_test, test_preds, zero_division=0)
        test_r = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_roc = roc_auc_score(y_test, test_probs)
        test_pr_auc = average_precision_score(y_test, test_probs)
        test_cm = confusion_matrix(y_test, test_preds).tolist()
        
        results[name] = {
            'val': {
                'precision': round(float(val_p), 4),
                'recall': round(float(val_r), 4),
                'f1': round(float(val_f1), 4),
                'roc_auc': round(float(val_roc), 4),
                'pr_auc': round(float(val_pr_auc), 4),
                'confusion_matrix': val_cm
            },
            'test': {
                'precision': round(float(test_p), 4),
                'recall': round(float(test_r), 4),
                'f1': round(float(test_f1), 4),
                'roc_auc': round(float(test_roc), 4),
                'pr_auc': round(float(test_pr_auc), 4),
                'confusion_matrix': test_cm
            }
        }
        print(f"  {name} -> Val ROC-AUC: {val_roc:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_p:.4f}, Val Recall: {val_r:.4f}")
        print(f"            Test ROC-AUC: {test_roc:.4f}, Test F1: {test_f1:.4f}, Test Precision: {test_p:.4f}, Test Recall: {test_r:.4f}")
        
    # Champion selection based on Val ROC-AUC
    champion_name = max(results.keys(), key=lambda k: results[k]['val']['roc_auc'])
    champion_model = fitted_models[champion_name]
    print(f"\nChampion Delay Model: {champion_name}")
    
    # Save Pipeline & Metrics
    pipeline = DelayPipeline(
        feature_engineer=fe,
        preprocessor=preprocessor,
        model=champion_model,
        feature_names=all_feature_names
    )
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    pipeline_path = os.path.join(MODELS_DIR, "delay_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved complete Delay Pipeline to {pipeline_path}")
    
    metrics_path = os.path.join(MODELS_DIR, "delay_metrics.json")
    export_metrics = {
        'champion_model': champion_name,
        'models': results,
        'feature_names': all_feature_names
    }
    with open(metrics_path, "w") as f:
        json.dump(export_metrics, f, indent=2)
    print(f"Saved Delay metrics to {metrics_path}")
    
    return pipeline, export_metrics

if __name__ == "__main__":
    train_and_evaluate_delay()

"""
Evaluation & Model Diagnostics for Smart Delivery ETA & Delay Engine.
Computes test set performance across all regression and classification models,
generates ROC curve coordinates, confusion matrices, and stores comprehensive evaluation summary.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))
from models_def import ETAPipeline, DelayPipeline

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def run_comprehensive_evaluation():
    print("Loading test dataset and trained pipelines...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    eta_pipe = joblib.load(os.path.join(MODELS_DIR, "eta_pipeline.joblib"))
    delay_pipe = joblib.load(os.path.join(MODELS_DIR, "delay_pipeline.joblib"))
    
    with open(os.path.join(MODELS_DIR, "eta_metrics.json"), "r") as f:
        eta_metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, "delay_metrics.json"), "r") as f:
        delay_metrics = json.load(f)

    # 1. Evaluate Champion ETA Pipeline on Test Set
    y_eta_true = test_df['actual_delivery_duration_min'].values
    y_eta_pred = eta_pipe.predict(test_df)
    
    eta_mae = float(mean_absolute_error(y_eta_true, y_eta_pred))
    eta_rmse = float(np.sqrt(mean_squared_error(y_eta_true, y_eta_pred)))
    eta_r2 = float(r2_score(y_eta_true, y_eta_pred))
    
    residuals = (y_eta_true - y_eta_pred).tolist()
    
    # Sample 400 points for lightweight plotting in Streamlit
    sample_indices = np.random.choice(len(y_eta_true), size=min(400, len(y_eta_true)), replace=False)
    actual_vs_pred_sample = [
        {'actual': round(float(y_eta_true[i]), 1), 'predicted': round(float(y_eta_pred[i]), 1)}
        for i in sample_indices
    ]

    # 2. Evaluate Champion Delay Pipeline on Test Set
    y_delay_true = test_df['delayed'].values
    y_delay_probs = delay_pipe.predict_proba(test_df)
    y_delay_pred = (y_delay_probs >= 0.5).astype(int)
    
    fpr, tpr, thresholds = roc_curve(y_delay_true, y_delay_probs)
    # Downsample ROC points for fast UI rendering
    stride = max(1, len(fpr) // 100)
    roc_data = [
        {'fpr': round(float(fpr[i]), 4), 'tpr': round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), stride)
    ]
    
    cm = confusion_matrix(y_delay_true, y_delay_pred).tolist()
    
    summary = {
        'eta_evaluation': {
            'champion': eta_metrics['champion_model'],
            'test_metrics': {
                'mae': round(eta_mae, 3),
                'rmse': round(eta_rmse, 3),
                'r2': round(eta_r2, 4)
            },
            'all_models': eta_metrics['models'],
            'sample_predictions': actual_vs_pred_sample,
            'prediction_intervals': eta_metrics['interval_quantiles']
        },
        'delay_evaluation': {
            'champion': delay_metrics['champion_model'],
            'test_metrics': {
                'precision': round(float(precision_score(y_delay_true, y_delay_pred)), 4),
                'recall': round(float(recall_score(y_delay_true, y_delay_pred)), 4),
                'f1': round(float(f1_score(y_delay_true, y_delay_pred)), 4),
                'roc_auc': round(float(roc_auc_score(y_delay_true, y_delay_probs)), 4),
                'pr_auc': round(float(average_precision_score(y_delay_true, y_delay_probs)), 4)
            },
            'all_models': delay_metrics['models'],
            'confusion_matrix': cm,
            'roc_curve': roc_data
        }
    }
    
    summary_path = os.path.join(MODELS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Comprehensive evaluation completed and saved to {summary_path}")
    print("\n--- Summary Performance on Test Set (8,250 unseen orders) ---")
    print(f"ETA Champion ({summary['eta_evaluation']['champion']}):")
    print(f"  MAE: {eta_mae:.2f} min | RMSE: {eta_rmse:.2f} min | R²: {eta_r2:.4f}")
    print(f"Delay Champion ({summary['delay_evaluation']['champion']}):")
    print(f"  ROC-AUC: {summary['delay_evaluation']['test_metrics']['roc_auc']:.4f} | F1: {summary['delay_evaluation']['test_metrics']['f1']:.4f} | Precision: {summary['delay_evaluation']['test_metrics']['precision']:.4f} | Recall: {summary['delay_evaluation']['test_metrics']['recall']:.4f}")
    
    return summary

if __name__ == "__main__":
    run_comprehensive_evaluation()

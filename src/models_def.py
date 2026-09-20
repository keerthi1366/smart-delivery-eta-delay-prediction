"""
Pipeline Definitions for Smart Delivery ETA & Delay Engine.
Encapsulates inference logic, empirical prediction intervals, and risk scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ETAPipeline:
    """End-to-end ETA inference pipeline bundling feature engineering, preprocessing, model, and interval quantiles."""
    def __init__(self, feature_engineer, preprocessor, model, interval_quantiles, feature_names):
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.model = model
        self.interval_quantiles = interval_quantiles
        self.feature_names = feature_names

    def predict(self, df_input: pd.DataFrame) -> np.ndarray:
        """Predict expected ETA in minutes."""
        df_feat = self.feature_engineer.transform(df_input)
        X_trans = self.preprocessor.transform(df_feat)
        preds = self.model.predict(X_trans)
        return np.maximum(10.0, np.round(preds, 1))

    def predict_with_interval(self, df_input: pd.DataFrame, confidence: int = 80) -> Dict[str, Any]:
        """Predict ETA with calibrated empirical prediction intervals."""
        preds = self.predict(df_input)
        pred_eta = float(preds[0]) if len(preds) == 1 else preds
        
        if confidence == 90:
            q_low = self.interval_quantiles.get('q05', -3.87)
            q_high = self.interval_quantiles.get('q95', 5.56)
        else: # 80% default
            q_low = self.interval_quantiles.get('q10', -3.30)
            q_high = self.interval_quantiles.get('q90', 4.04)
            
        if isinstance(pred_eta, (float, int, np.floating)):
            low = max(8.0, round(float(pred_eta) + q_low, 1))
            high = round(float(pred_eta) + q_high, 1)
            return {
                'predicted_eta': round(float(pred_eta), 1),
                'interval_lower': low,
                'interval_upper': high,
                'range_str': f"{int(round(low))}–{int(round(high))} min",
                'confidence_pct': confidence
            }
        else:
            return {
                'predicted_eta': pred_eta,
                'interval_lower': np.maximum(8.0, np.round(pred_eta + q_low, 1)),
                'interval_upper': np.round(pred_eta + q_high, 1),
                'confidence_pct': confidence
            }


class DelayPipeline:
    """End-to-end Delay inference pipeline bundling feature engineering, preprocessing, classifier, and risk scoring."""
    def __init__(self, feature_engineer, preprocessor, model, feature_names, risk_thresholds=None):
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.model = model
        self.feature_names = feature_names
        self.risk_thresholds = risk_thresholds or {'low': 25, 'medium': 50, 'high': 75}

    def predict_proba(self, df_input: pd.DataFrame) -> np.ndarray:
        """Predict delay probability between 0.0 and 1.0."""
        df_feat = self.feature_engineer.transform(df_input)
        X_trans = self.preprocessor.transform(df_feat)
        return self.model.predict_proba(X_trans)[:, 1]

    def predict_risk(self, df_input: pd.DataFrame) -> Dict[str, Any]:
        """Convert delay probability into risk score, category, and styling."""
        probs = self.predict_proba(df_input)
        prob = float(probs[0]) if len(probs) == 1 else probs
        
        def _get_tier(p_val: float):
            score = int(round(p_val * 100))
            if score < self.risk_thresholds['low']:
                tier = 'LOW'
                color = '#28a745'
            elif score < self.risk_thresholds['medium']:
                tier = 'MEDIUM'
                color = '#ffc107'
            elif score < self.risk_thresholds['high']:
                tier = 'HIGH'
                color = '#fd7e14'
            else:
                tier = 'CRITICAL'
                color = '#dc3545'
            return score, tier, color

        if isinstance(prob, (float, int, np.floating)):
            score, tier, color = _get_tier(float(prob))
            return {
                'probability': round(float(prob), 4),
                'probability_pct': round(float(prob) * 100, 1),
                'risk_score': score,
                'risk_level': tier,
                'color': color
            }
        else:
            scores = [int(round(p * 100)) for p in prob]
            return {
                'probabilities': np.round(prob, 4),
                'risk_scores': np.array(scores)
            }

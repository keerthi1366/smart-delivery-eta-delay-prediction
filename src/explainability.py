"""
Explainability Module for Smart Delivery ETA & Delay Engine.
Computes global and local feature contributions using SHAP (SHapley Additive exPlanations).
Identifies the top contributing factors for individual order predictions in real-time.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Any

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))
from models_def import ETAPipeline, DelayPipeline

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

class DeliveryExplainer:
    """Manages SHAP explanations for delivery ETA and delay predictions."""
    
    def __init__(self, eta_pipeline_path: str = None, delay_pipeline_path: str = None):
        if eta_pipeline_path is None:
            eta_pipeline_path = os.path.join(MODELS_DIR, "eta_pipeline.joblib")
        if delay_pipeline_path is None:
            delay_pipeline_path = os.path.join(MODELS_DIR, "delay_pipeline.joblib")
            
        self.eta_pipe = joblib.load(eta_pipeline_path)
        self.delay_pipe = joblib.load(delay_pipeline_path)
        
        self.feature_names = self.eta_pipe.feature_names
        
        # Initialize Explainer for ETA (XGBoost regressor or RF)
        self.eta_model = self.eta_pipe.model
        if hasattr(self.eta_model, 'get_booster'):
            self.eta_explainer = shap.TreeExplainer(self.eta_model)
        else:
            self.eta_explainer = None

    def explain_order(self, df_input: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Compute local SHAP contribution for a single order input.
        Returns top factors pushing ETA up (slower) or down (faster).
        """
        # Feature transformation
        df_feat = self.eta_pipe.feature_engineer.transform(df_input)
        X_trans = self.eta_pipe.preprocessor.transform(df_feat)
        
        if self.eta_explainer is not None:
            shap_values = self.eta_explainer.shap_values(X_trans)
            if isinstance(shap_values, list):
                shap_vals = shap_values[0][0]
            elif shap_values.ndim > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
        else:
            # Fallback to feature importance weighting
            importances = getattr(self.eta_model, 'feature_importances_', np.ones(len(self.feature_names)))
            shap_vals = (X_trans[0] - np.mean(X_trans, axis=0)) * importances

        # Map to human-friendly feature names and aggregate one-hot categories
        contributions = []
        for name, val in zip(self.feature_names, shap_vals):
            clean_name = self._beautify_feature_name(name)
            contributions.append({
                'feature': clean_name,
                'raw_name': name,
                'shap_value': round(float(val), 2),
                'direction': 'Increases ETA (+Delay)' if val > 0 else 'Decreases ETA (Faster)'
            })
            
        # Sort by absolute impact
        contributions_sorted = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        top_factors = contributions_sorted[:top_k]
        
        return {
            'top_factors': top_factors,
            'all_contributions': contributions_sorted
        }

    def _beautify_feature_name(self, name: str) -> str:
        """Convert technical feature column names into executive friendly labels."""
        mapping = {
            'distance_km': 'Delivery Distance (km)',
            'restaurant_prep_time': 'Restaurant Kitchen Prep Time',
            'orders_last_30_min': 'Recent Order Surge (Demand)',
            'active_delivery_partners': 'Active Couriers Online (Supply)',
            'demand_supply_ratio': 'Fleet Demand/Supply Pressure',
            'traffic_score': 'Traffic Congestion Level',
            'weather_severity': 'Adverse Weather Impact',
            'distance_x_traffic': 'Distance × Traffic Combined Burden',
            'prep_x_item_count': 'Kitchen Volume Burden (Prep × Items)',
            'item_count': 'Order Item Count',
            'order_value': 'Cart Value',
            'precipitation': 'Rain Precipitation',
            'temperature': 'Ambient Temperature',
            'hist_rest_prep_time': 'Historical Restaurant Prep Speed',
            'hist_rest_delay_rate': 'Historical Restaurant Delay Rate',
            'delivery_partner_experience_months': 'Courier Experience (Months)'
        }
        if name in mapping:
            return mapping[name]
        if name.startswith('cat__'):
            cleaned = name.replace('cat__', '').replace('_', ' ').title()
            return f"{cleaned}"
        return name.replace('_', ' ').title()

    def get_global_feature_importance(self, top_n: int = 12) -> pd.DataFrame:
        """Fetch global feature importances from champion ETA model."""
        if hasattr(self.eta_model, 'feature_importances_'):
            importances = self.eta_model.feature_importances_
        elif hasattr(self.eta_model, 'coef_'):
            importances = np.abs(self.eta_model.coef_)
        else:
            importances = np.ones(len(self.feature_names))
            
        df_imp = pd.DataFrame({
            'Feature': [self._beautify_feature_name(f) for f in self.feature_names],
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Normalize to 100%
        df_imp['Relative Importance (%)'] = np.round(100.0 * df_imp['Importance'] / df_imp['Importance'].sum(), 2)
        return df_imp

if __name__ == "__main__":
    explainer = DeliveryExplainer()
    df_sample = pd.DataFrame([{
        'restaurant_id': 'REST_005',
        'customer_id': 'CUST_0012',
        'delivery_partner_id': 'DP_015',
        'order_time': '2026-09-20 19:30:00',
        'day_of_week': 'Friday',
        'distance_km': 14.5,
        'restaurant_prep_time': 38.0,
        'order_size': 'Large',
        'item_count': 8,
        'order_value': 85.0,
        'traffic_level': 'Jam',
        'weather': 'Stormy',
        'temperature': 22.0,
        'precipitation': 24.0,
        'active_delivery_partners': 35,
        'orders_last_30_min': 140,
        'peak_hour': 1,
        'delivery_partner_experience': 'Novice',
        'delivery_partner_experience_months': 2
    }])
    
    explanation = explainer.explain_order(df_sample)
    print("Top contributing factors for sample order:")
    for f in explanation['top_factors']:
        print(f"  * {f['feature']}: {f['shap_value']:+0.2f} min ({f['direction']})")
    
    global_imp = explainer.get_global_feature_importance()
    print("\nGlobal Feature Importance (Top 5):")
    print(global_imp.head(5))

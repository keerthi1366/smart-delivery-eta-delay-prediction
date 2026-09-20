"""
Tests for Saved ML Pipelines (ETA Regression & Delay Classification).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from models_def import ETAPipeline, DelayPipeline

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

@pytest.fixture
def sample_input():
    return pd.DataFrame([{
        'restaurant_id': 'REST_005',
        'customer_id': 'CUST_0025',
        'delivery_partner_id': 'DP_010',
        'order_time': '2026-06-20 20:15:00',
        'day_of_week': 'Saturday',
        'distance_km': 7.2,
        'restaurant_prep_time': 24.0,
        'order_size': 'Medium',
        'item_count': 5,
        'order_value': 52.0,
        'traffic_level': 'Medium',
        'weather': 'Clear',
        'temperature': 28.0,
        'precipitation': 0.0,
        'active_delivery_partners': 75,
        'orders_last_30_min': 80,
        'peak_hour': 1,
        'delivery_partner_experience': 'Intermediate',
        'delivery_partner_experience_months': 10
    }])

def test_eta_pipeline(sample_input):
    eta_path = os.path.join(MODELS_DIR, "eta_pipeline.joblib")
    assert os.path.exists(eta_path), "eta_pipeline.joblib missing"
    pipe = joblib.load(eta_path)
    
    # Predict ETA
    preds = pipe.predict(sample_input)
    assert len(preds) == 1
    assert preds[0] >= 10.0, "ETA must be at least 10 minutes"
    
    # Predict Interval
    res = pipe.predict_with_interval(sample_input, confidence=80)
    assert 'predicted_eta' in res
    assert 'interval_lower' in res
    assert 'interval_upper' in res
    assert res['interval_lower'] <= res['predicted_eta'] <= res['interval_upper']
    assert 'min' in res['range_str']

def test_delay_pipeline(sample_input):
    delay_path = os.path.join(MODELS_DIR, "delay_pipeline.joblib")
    assert os.path.exists(delay_path), "delay_pipeline.joblib missing"
    pipe = joblib.load(delay_path)
    
    # Predict Probability
    prob = pipe.predict_proba(sample_input)
    assert len(prob) == 1
    assert 0.0 <= prob[0] <= 1.0
    
    # Predict Risk
    risk = pipe.predict_risk(sample_input)
    assert 'probability_pct' in risk
    assert 0 <= risk['risk_score'] <= 100
    assert risk['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    assert risk['color'].startswith('#')

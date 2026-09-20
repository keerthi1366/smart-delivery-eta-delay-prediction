"""
Tests for Feature Engineering and Pipeline Transformers.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from features import DeliveryFeatureEngineer, build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def test_feature_engineering_transforms():
    fe = DeliveryFeatureEngineer()
    df_sample = pd.DataFrame([{
        'restaurant_id': 'REST_001',
        'customer_id': 'CUST_001',
        'delivery_partner_id': 'DP_001',
        'order_time': '2026-05-15 19:30:00',
        'distance_km': 6.5,
        'restaurant_prep_time': 25.0,
        'order_size': 'Medium',
        'item_count': 4,
        'order_value': 45.0,
        'traffic_level': 'High',
        'weather': 'Rainy',
        'temperature': 24.0,
        'precipitation': 6.0,
        'active_delivery_partners': 50,
        'orders_last_30_min': 70,
        'delivery_partner_experience': 'Intermediate',
        'delivery_partner_experience_months': 8
    }])
    
    transformed = fe.transform(df_sample)
    
    # Check created features
    assert 'hour' in transformed.columns
    assert transformed['hour'].iloc[0] == 19
    assert transformed['peak_hour'].iloc[0] == 1
    assert transformed['traffic_score'].iloc[0] == 3 # High -> 3
    assert transformed['weather_severity'].iloc[0] == 3 # Rainy -> 3
    assert transformed['distance_x_traffic'].iloc[0] == 6.5 * 3
    assert transformed['distance_bucket'].iloc[0] == 'Medium'
    assert transformed['demand_supply_ratio'].iloc[0] == pytest.approx(70 / 51, rel=1e-3)

def test_preprocessor_shape_and_no_nans():
    fe = DeliveryFeatureEngineer()
    df_sample = pd.DataFrame([{
        'restaurant_id': 'REST_002',
        'customer_id': 'CUST_002',
        'delivery_partner_id': 'DP_002',
        'order_time': '2026-05-15 12:15:00',
        'distance_km': 3.2,
        'restaurant_prep_time': 18.0,
        'order_size': 'Small',
        'item_count': 2,
        'order_value': 22.0,
        'traffic_level': 'Low',
        'weather': 'Clear',
        'temperature': 28.0,
        'precipitation': 0.0,
        'active_delivery_partners': 40,
        'orders_last_30_min': 35,
        'delivery_partner_experience': 'Experienced',
        'delivery_partner_experience_months': 24
    }])
    
    transformed = fe.transform(df_sample)
    preprocessor = build_preprocessor()
    X_mat = preprocessor.fit_transform(transformed)
    
    assert X_mat.shape[0] == 1
    assert not np.isnan(X_mat).any(), "Preprocessor output contains NaNs"

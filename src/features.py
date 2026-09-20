"""
Feature Engineering Pipeline for Smart Delivery Engine.
Extracts pre-delivery features, creates interaction terms, distance buckets,
and standardizes numeric/categorical columns without data leakage.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Dict, Any

TRAFFIC_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}
WEATHER_SEVERITY_MAP = {'Clear': 0, 'Windy': 1, 'Foggy': 2, 'Rainy': 3, 'Stormy': 4}
EXP_MAP = {'Novice': 1, 'Intermediate': 2, 'Experienced': 3}

class DeliveryFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to engineer pre-delivery operational features."""
    
    def __init__(self, baselines_path: str = None):
        self.baselines_path = baselines_path
        self.baselines = None
        if baselines_path and os.path.exists(baselines_path):
            with open(baselines_path, 'r') as f:
                self.baselines = json.load(f)
                
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Temporal features
        if 'order_time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['order_time']):
                df['order_time'] = pd.to_datetime(df['order_time'])
            df['hour'] = df['order_time'].dt.hour
            df['day_of_week'] = df['order_time'].dt.day_name()
            df['is_weekend'] = df['order_time'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        else:
            if 'hour' not in df.columns:
                df['hour'] = 19
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = 'Friday'
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = 1 if df['day_of_week'].iloc[0] in ['Saturday', 'Sunday'] else 0
                
        if 'peak_hour' not in df.columns:
            df['peak_hour'] = df['hour'].apply(lambda h: 1 if h in [12, 13, 19, 20, 21] else 0)

        # 2. Distance buckets
        df['distance_bucket'] = pd.cut(
            df['distance_km'],
            bins=[-np.inf, 3.0, 7.0, 12.0, np.inf],
            labels=['Short', 'Medium', 'Long', 'Extended']
        ).astype(str)

        # 3. Supply & Demand
        df['demand_supply_ratio'] = df['orders_last_30_min'] / (df['active_delivery_partners'] + 1.0)

        # 4. Ordinal encodings for interaction terms
        traffic_score = df['traffic_level'].map(TRAFFIC_MAP).fillna(2)
        weather_severity = df['weather'].map(WEATHER_SEVERITY_MAP).fillna(0)
        
        df['traffic_score'] = traffic_score
        df['weather_severity'] = weather_severity

        # 5. Domain Interaction features
        df['distance_x_traffic'] = df['distance_km'] * traffic_score
        df['prep_x_item_count'] = df['restaurant_prep_time'] * df['item_count']
        df['adverse_weather_x_traffic'] = weather_severity * traffic_score
        df['distance_x_weather'] = df['distance_km'] * (1.0 + 0.15 * weather_severity)

        # 6. Historical restaurant & partner metrics resolution (if not present)
        if self.baselines:
            gb = self.baselines.get('global_baselines', {})
            rests = self.baselines.get('restaurants', {})
            couriers = self.baselines.get('couriers', {})
            
            if 'hist_rest_prep_time' not in df.columns:
                df['hist_rest_prep_time'] = df['restaurant_id'].map(
                    lambda r: rests.get(r, {}).get('train_rest_avg_prep', gb.get('avg_prep_time', 20.0))
                )
            if 'hist_rest_delivery_time' not in df.columns:
                df['hist_rest_delivery_time'] = df['restaurant_id'].map(
                    lambda r: rests.get(r, {}).get('train_rest_avg_delivery', gb.get('avg_delivery_duration', 45.0))
                )
            if 'hist_rest_delay_rate' not in df.columns:
                df['hist_rest_delay_rate'] = df['restaurant_id'].map(
                    lambda r: rests.get(r, {}).get('train_rest_delay_rate', gb.get('delay_rate', 0.20))
                )
            if 'hist_courier_delivery_time' not in df.columns:
                df['hist_courier_delivery_time'] = df['delivery_partner_id'].map(
                    lambda c: couriers.get(c, {}).get('train_courier_avg_delivery', gb.get('avg_delivery_duration', 45.0))
                )
            if 'hist_courier_delay_rate' not in df.columns:
                df['hist_courier_delay_rate'] = df['delivery_partner_id'].map(
                    lambda c: couriers.get(c, {}).get('train_courier_delay_rate', gb.get('delay_rate', 0.20))
                )
        else:
            # Fallback default values if baselines file not available
            for col in ['hist_rest_prep_time', 'hist_rest_delivery_time', 'hist_rest_delay_rate',
                        'hist_courier_delivery_time', 'hist_courier_delay_rate']:
                if col not in df.columns:
                    df[col] = 0.0

        return df

NUMERIC_FEATURES = [
    'distance_km',
    'restaurant_prep_time',
    'item_count',
    'order_value',
    'temperature',
    'precipitation',
    'active_delivery_partners',
    'orders_last_30_min',
    'demand_supply_ratio',
    'traffic_score',
    'weather_severity',
    'distance_x_traffic',
    'prep_x_item_count',
    'adverse_weather_x_traffic',
    'distance_x_weather',
    'hist_rest_prep_time',
    'hist_rest_delivery_time',
    'hist_rest_delay_rate',
    'delivery_partner_experience_months'
]

CATEGORICAL_FEATURES = [
    'order_size',
    'traffic_level',
    'weather',
    'peak_hour',
    'is_weekend',
    'day_of_week',
    'distance_bucket',
    'delivery_partner_experience'
]

def build_preprocessor() -> ColumnTransformer:
    """Build Sklearn ColumnTransformer for scaling numeric and one-hot encoding categorical."""
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERIC_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

if __name__ == "__main__":
    from data_cleaning import PROCESSED_DIR, MODELS_DIR
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    baselines_path = os.path.join(MODELS_DIR, "historical_baselines.json")
    
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        fe = DeliveryFeatureEngineer(baselines_path=baselines_path)
        df_feat = fe.transform(df_train)
        print(f"Engineered features shape: {df_feat.shape}")
        prep = build_preprocessor()
        X_trans = prep.fit_transform(df_feat)
        print(f"Preprocessed matrix shape: {X_trans.shape}")
        print("Feature engineering successfully verified.")

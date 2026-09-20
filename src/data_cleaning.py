"""
Data Cleaning & Time-based Train/Validation/Test Split for Smart Delivery Engine.
Ensures zero data leakage by splitting chronologically and computing historical features
strictly on past training observations.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "delivery_orders.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def clean_and_split_data(
    raw_path: str = RAW_DATA_PATH,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Clean dataset and perform strict chronological time-based split.
    Calculates historical restaurant and partner baselines strictly on training set.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)
    
    # 1. Type validation and chronological sorting
    df['order_time'] = pd.to_datetime(df['order_time'])
    df = df.sort_values('order_time').reset_index(drop=True)
    
    # Check nulls and duplicates
    df = df.drop_duplicates(subset=['order_id'])
    
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # 2. Compute historical baselines strictly from training set (Zero Lookahead Bias)
    # Global fallbacks for any unseen restaurants/couriers
    global_avg_prep = float(train_df['restaurant_prep_time'].mean())
    global_avg_delivery = float(train_df['actual_delivery_duration_min'].mean())
    global_delay_rate = float(train_df['delayed'].mean())
    
    rest_stats = train_df.groupby('restaurant_id').agg(
        train_rest_avg_prep=('restaurant_prep_time', 'mean'),
        train_rest_avg_delivery=('actual_delivery_duration_min', 'mean'),
        train_rest_delay_rate=('delayed', 'mean'),
        train_rest_orders=('order_id', 'count')
    ).reset_index()
    
    courier_stats = train_df.groupby('delivery_partner_id').agg(
        train_courier_avg_delivery=('actual_delivery_duration_min', 'mean'),
        train_courier_delay_rate=('delayed', 'mean'),
        train_courier_orders=('order_id', 'count')
    ).reset_index()
    
    rest_lookup = rest_stats.set_index('restaurant_id').to_dict(orient='index')
    courier_lookup = courier_stats.set_index('delivery_partner_id').to_dict(orient='index')
    
    metadata = {
        'global_baselines': {
            'avg_prep_time': round(global_avg_prep, 2),
            'avg_delivery_duration': round(global_avg_delivery, 2),
            'delay_rate': round(global_delay_rate, 4)
        },
        'restaurants': {k: {m: round(float(v[m]), 4) for m in v} for k, v in rest_lookup.items()},
        'couriers': {k: {m: round(float(v[m]), 4) for m in v} for k, v in courier_lookup.items()}
    }
    
    # Map back to train, val, test sets
    for split_df in [train_df, val_df, test_df]:
        split_df['hist_rest_prep_time'] = split_df['restaurant_id'].map(
            lambda r: rest_lookup.get(r, {}).get('train_rest_avg_prep', global_avg_prep)
        )
        split_df['hist_rest_delivery_time'] = split_df['restaurant_id'].map(
            lambda r: rest_lookup.get(r, {}).get('train_rest_avg_delivery', global_avg_delivery)
        )
        split_df['hist_rest_delay_rate'] = split_df['restaurant_id'].map(
            lambda r: rest_lookup.get(r, {}).get('train_rest_delay_rate', global_delay_rate)
        )
        split_df['hist_courier_delivery_time'] = split_df['delivery_partner_id'].map(
            lambda c: courier_lookup.get(c, {}).get('train_courier_avg_delivery', global_avg_delivery)
        )
        split_df['hist_courier_delay_rate'] = split_df['delivery_partner_id'].map(
            lambda c: courier_lookup.get(c, {}).get('train_courier_delay_rate', global_delay_rate)
        )
    
    if save_splits:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
        
        with open(os.path.join(MODELS_DIR, "historical_baselines.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Splits successfully created and saved:")
        print(f"  Train: {len(train_df):,} orders ({train_df['order_time'].min()} to {train_df['order_time'].max()})")
        print(f"  Val:   {len(val_df):,} orders ({val_df['order_time'].min()} to {val_df['order_time'].max()})")
        print(f"  Test:  {len(test_df):,} orders ({test_df['order_time'].min()} to {test_df['order_time'].max()})")
        
    return train_df, val_df, test_df, metadata

if __name__ == "__main__":
    clean_and_split_data()

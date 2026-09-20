"""
Tests for Data Cleaning and Time-based Split.
"""

import os
import sys
import json
import pandas as pd
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def test_processed_files_exist():
    assert os.path.exists(os.path.join(PROCESSED_DIR, "train.csv")), "train.csv does not exist"
    assert os.path.exists(os.path.join(PROCESSED_DIR, "val.csv")), "val.csv does not exist"
    assert os.path.exists(os.path.join(PROCESSED_DIR, "test.csv")), "test.csv does not exist"

def test_chronological_time_split_no_leakage():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    train_max = pd.to_datetime(train_df['order_time']).max()
    val_min = pd.to_datetime(val_df['order_time']).min()
    val_max = pd.to_datetime(val_df['order_time']).max()
    test_min = pd.to_datetime(test_df['order_time']).min()

    assert train_max <= val_min, f"Temporal leakage: train_max {train_max} > val_min {val_min}"
    assert val_max <= test_min, f"Temporal leakage: val_max {val_max} > test_min {test_min}"

def test_historical_baselines_metadata():
    baselines_path = os.path.join(MODELS_DIR, "historical_baselines.json")
    assert os.path.exists(baselines_path), "historical_baselines.json missing"
    with open(baselines_path, "r") as f:
        data = json.load(f)
    assert "global_baselines" in data
    assert "restaurants" in data
    assert data["global_baselines"]["avg_prep_time"] > 0

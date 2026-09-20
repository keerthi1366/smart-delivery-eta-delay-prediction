"""
Tests for SQLite Database and Analytical Queries.
"""

import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from db_manager import DeliveryDatabase, DEFAULT_DB_PATH

def test_database_queries():
    assert os.path.exists(DEFAULT_DB_PATH), "Database file does not exist"
    db = DeliveryDatabase(DEFAULT_DB_PATH)
    
    # 1. KPIs
    kpis = db.get_kpis()
    assert kpis['total_orders'] >= 50000
    assert kpis['avg_delivery_time_min'] > 0
    assert 0 <= kpis['delay_rate_pct'] <= 100
    
    # 2. Restaurant performance
    df_rest = db.get_restaurant_performance(limit=5)
    assert len(df_rest) == 5
    assert 'avg_prep_time_min' in df_rest.columns
    assert 'delay_rate_pct' in df_rest.columns
    
    # 3. Traffic impact
    df_traffic = db.get_traffic_impact()
    assert len(df_traffic) == 4 # Low, Medium, High, Jam
    assert 'traffic_level' in df_traffic.columns
    
    # 4. Weather impact
    df_weather = db.get_weather_impact()
    assert len(df_weather) >= 4
    assert 'weather' in df_weather.columns
    
    # 5. Partner performance
    df_partner = db.get_partner_performance()
    assert len(df_partner) == 3 # Novice, Intermediate, Experienced
    assert 'delivery_partner_experience' in df_partner.columns

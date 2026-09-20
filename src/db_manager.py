"""
Database Manager for Smart Delivery ETA & Delay Prediction Engine.
Handles SQLite database initialization, indexing, and high-performance SQL analytics.
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "delivery_orders.db")
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "delivery_orders.csv")

class DeliveryDatabase:
    """Manages SQLite connection and analytical queries for delivery orders."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_db(self, csv_path: str = DEFAULT_CSV_PATH, force_reload: bool = False):
        """Create tables, load dataset, and build indexes."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Source CSV not found at {csv_path}. Please generate data first.")
        
        # Check if table already has records
        if os.path.exists(self.db_path) and not force_reload:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='delivery_orders';")
                if cur.fetchone():
                    cur.execute("SELECT COUNT(*) FROM delivery_orders;")
                    count = cur.fetchone()[0]
                    if count > 0:
                        print(f"Database already initialized with {count:,} orders at {self.db_path}")
                        return

        print(f"Loading data from {csv_path} into SQLite database {self.db_path}...")
        df = pd.read_csv(csv_path)
        
        with self.get_connection() as conn:
            # Write dataframe to SQL table
            df.to_sql('delivery_orders', conn, if_exists='replace', index=False)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_order_time ON delivery_orders(order_time);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_restaurant ON delivery_orders(restaurant_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_partner ON delivery_orders(delivery_partner_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_delayed ON delivery_orders(delayed);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_weather ON delivery_orders(traffic_level, weather);")
            conn.commit()
            
        print(f"Successfully loaded {len(df):,} records and built indexes.")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute arbitrary SQL query and return a pandas DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ----------------- Specialized Analytics Methods -----------------

    def get_kpis(self) -> Dict[str, Any]:
        """Fetch overall KPI metrics."""
        query = """
        SELECT
            COUNT(*) AS total_orders,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_time_min,
            ROUND(AVG(promised_delivery_duration_min), 2) AS avg_promised_time_min,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders;
        """
        df = self.execute_query(query)
        return df.iloc[0].to_dict()

    def get_restaurant_performance(self, limit: int = 20) -> pd.DataFrame:
        """Fetch restaurant performance leaderboard."""
        query = f"""
        SELECT
            restaurant_id,
            COUNT(*) AS total_orders,
            ROUND(AVG(restaurant_prep_time), 2) AS avg_prep_time_min,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_time_min,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_min,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY restaurant_id
        ORDER BY total_orders DESC
        LIMIT {limit};
        """
        return self.execute_query(query)

    def get_peak_hour_delays(self) -> pd.DataFrame:
        """Fetch comparison between Peak-Hour and Off-Peak delays."""
        query = """
        SELECT
            CASE WHEN peak_hour = 1 THEN 'Peak Hour' ELSE 'Off-Peak' END AS period_type,
            COUNT(*) AS total_orders,
            ROUND(AVG(restaurant_prep_time), 2) AS avg_prep_min,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_min,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY peak_hour
        ORDER BY peak_hour DESC;
        """
        return self.execute_query(query)

    def get_traffic_impact(self) -> pd.DataFrame:
        """Fetch traffic level delivery time and delay stats."""
        query = """
        SELECT
            traffic_level,
            COUNT(*) AS total_orders,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_min,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY traffic_level
        ORDER BY avg_delivery_min ASC;
        """
        return self.execute_query(query)

    def get_weather_impact(self) -> pd.DataFrame:
        """Fetch weather condition stats."""
        query = """
        SELECT
            weather,
            COUNT(*) AS total_orders,
            ROUND(AVG(precipitation), 2) AS avg_precip_mm,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_min,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY weather
        ORDER BY delay_rate_pct DESC;
        """
        return self.execute_query(query)

    def get_distance_vs_delivery_time(self) -> pd.DataFrame:
        """Fetch distance bucket breakdown."""
        query = """
        SELECT
            CASE
                WHEN distance_km < 3.0 THEN 'Short (< 3 km)'
                WHEN distance_km < 7.0 THEN 'Medium (3-7 km)'
                WHEN distance_km < 12.0 THEN 'Long (7-12 km)'
                ELSE 'Extended (> 12 km)'
            END AS distance_category,
            COUNT(*) AS total_orders,
            ROUND(AVG(distance_km), 2) AS avg_distance_km,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY distance_category
        ORDER BY avg_distance_km ASC;
        """
        return self.execute_query(query)

    def get_partner_performance(self) -> pd.DataFrame:
        """Fetch delivery partner performance grouped by experience tier."""
        query = """
        SELECT
            delivery_partner_experience,
            COUNT(*) AS total_deliveries,
            ROUND(AVG(delivery_partner_experience_months), 1) AS avg_experience_months,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY delivery_partner_experience
        ORDER BY avg_delivery_min ASC;
        """
        return self.execute_query(query)

    def get_monthly_trends(self) -> pd.DataFrame:
        """Fetch monthly volume, average delivery time, and delay rate trends."""
        query = """
        SELECT
            SUBSTR(order_time, 1, 7) AS month,
            COUNT(*) AS order_volume,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_time,
            ROUND(AVG(delay_minutes), 2) AS avg_delay_time,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY month
        ORDER BY month ASC;
        """
        return self.execute_query(query)

    def get_hourly_delay_distribution(self) -> pd.DataFrame:
        """Fetch delay distribution across all 24 hours of the day."""
        query = """
        SELECT
            CAST(SUBSTR(order_time, 12, 2) AS INTEGER) AS hour_of_day,
            COUNT(*) AS total_orders,
            ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
            ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
        FROM delivery_orders
        GROUP BY hour_of_day
        ORDER BY hour_of_day ASC;
        """
        return self.execute_query(query)

if __name__ == "__main__":
    db = DeliveryDatabase()
    db.initialize_db(force_reload=True)
    kpis = db.get_kpis()
    print("Database Initialized Successfully. KPI Summary:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

-- SQLite Schema and Analytics Queries for Smart Delivery ETA & Delay Prediction Engine

-- 1. Table Schema
CREATE TABLE IF NOT EXISTS delivery_orders (
    order_id TEXT PRIMARY KEY,
    restaurant_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    delivery_partner_id TEXT NOT NULL,
    order_time TEXT NOT NULL,
    day_of_week TEXT NOT NULL,
    distance_km REAL NOT NULL,
    restaurant_prep_time REAL NOT NULL,
    restaurant_historical_delivery_time REAL NOT NULL,
    restaurant_historical_delay_rate REAL NOT NULL,
    order_size TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    order_value REAL NOT NULL,
    traffic_level TEXT NOT NULL,
    weather TEXT NOT NULL,
    temperature REAL NOT NULL,
    precipitation REAL NOT NULL,
    active_delivery_partners INTEGER NOT NULL,
    orders_last_30_min INTEGER NOT NULL,
    peak_hour INTEGER NOT NULL,
    delivery_partner_experience TEXT NOT NULL,
    delivery_partner_experience_months INTEGER NOT NULL,
    promised_delivery_time TEXT NOT NULL,
    actual_delivery_time TEXT NOT NULL,
    actual_delivery_duration_min REAL NOT NULL,
    promised_delivery_duration_min REAL NOT NULL,
    delay_minutes REAL NOT NULL,
    delayed INTEGER NOT NULL
);

-- Indexes for fast analytics
CREATE INDEX IF NOT EXISTS idx_order_time ON delivery_orders(order_time);
CREATE INDEX IF NOT EXISTS idx_restaurant ON delivery_orders(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_partner ON delivery_orders(delivery_partner_id);
CREATE INDEX IF NOT EXISTS idx_delayed ON delivery_orders(delayed);
CREATE INDEX IF NOT EXISTS idx_traffic_weather ON delivery_orders(traffic_level, weather);

-- -------------------------------------------------------------
-- ANALYTICS QUERIES
-- -------------------------------------------------------------

-- Query 1: Overall KPIs (Average Delivery Time & Delay Rate)
-- SELECT
--     COUNT(*) AS total_orders,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_time_min,
--     ROUND(AVG(promised_delivery_duration_min), 2) AS avg_promised_time_min,
--     ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders;

-- Query 2: Restaurant Performance (Volume, Prep Time, Delivery Time, Delay Rate)
-- SELECT
--     restaurant_id,
--     COUNT(*) AS total_orders,
--     ROUND(AVG(restaurant_prep_time), 2) AS avg_prep_time_min,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_time_min,
--     ROUND(AVG(delay_minutes), 2) AS avg_delay_min,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY restaurant_id
-- ORDER BY total_orders DESC;

-- Query 3: Peak-Hour vs Non-Peak Delays
-- SELECT
--     CASE WHEN peak_hour = 1 THEN 'Peak Hour' ELSE 'Off-Peak' END AS period_type,
--     COUNT(*) AS total_orders,
--     ROUND(AVG(restaurant_prep_time), 2) AS avg_prep_min,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY peak_hour;

-- Query 4: Traffic Level Impact
-- SELECT
--     traffic_level,
--     COUNT(*) AS total_orders,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
--     ROUND(AVG(delay_minutes), 2) AS avg_delay_min,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY traffic_level
-- ORDER BY avg_delivery_min ASC;

-- Query 5: Weather Condition Impact
-- SELECT
--     weather,
--     COUNT(*) AS total_orders,
--     ROUND(AVG(precipitation), 2) AS avg_precip_mm,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
--     ROUND(AVG(delay_minutes), 2) AS avg_delay_min,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY weather
-- ORDER BY delay_rate_pct DESC;

-- Query 6: Distance Bucket vs Delivery Time
-- SELECT
--     CASE
--         WHEN distance_km < 3.0 THEN 'Short (< 3 km)'
--         WHEN distance_km < 7.0 THEN 'Medium (3-7 km)'
--         WHEN distance_km < 12.0 THEN 'Long (7-12 km)'
--         ELSE 'Extended (> 12 km)'
--     END AS distance_category,
--     COUNT(*) AS total_orders,
--     ROUND(AVG(distance_km), 2) AS avg_distance_km,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY distance_category
-- ORDER BY avg_distance_km ASC;

-- Query 7: Delivery Partner Performance by Experience
-- SELECT
--     delivery_partner_experience,
--     COUNT(*) AS total_deliveries,
--     ROUND(AVG(delivery_partner_experience_months), 1) AS avg_experience_months,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY delivery_partner_experience
-- ORDER BY avg_delivery_min ASC;

-- Query 8: Monthly Trends
-- SELECT
--     SUBSTR(order_time, 1, 7) AS month,
--     COUNT(*) AS order_volume,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_time,
--     ROUND(AVG(delay_minutes), 2) AS avg_delay_time,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY month
-- ORDER BY month ASC;

-- Query 9: Hourly Delay Distribution
-- SELECT
--     CAST(SUBSTR(order_time, 12, 2) AS INTEGER) AS hour_of_day,
--     COUNT(*) AS total_orders,
--     ROUND(AVG(actual_delivery_duration_min), 2) AS avg_delivery_min,
--     ROUND(AVG(delayed) * 100.0, 2) AS delay_rate_pct
-- FROM delivery_orders
-- GROUP BY hour_of_day
-- ORDER BY hour_of_day ASC;

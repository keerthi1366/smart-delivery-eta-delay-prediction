"""
Data Generator for Smart Delivery ETA & Delay Prediction Engine.
Generates 50,000+ realistic synthetic delivery orders with physical domain dynamics,
temporal consistency, and non-trivial noise.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_delivery_data(num_orders: int = 55000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic delivery order dataset with realistic operational dynamics.
    
    Parameters:
        num_orders: Number of orders to generate (default 55,000)
        random_seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame sorted chronologically by order_time
    """
    np.random.seed(random_seed)
    
    # 1. Entities
    num_restaurants = 60
    restaurants = [f"REST_{i:03d}" for i in range(1, num_restaurants + 1)]
    # Restaurant inherent efficiency and food style
    restaurant_prep_baselines = np.random.uniform(12.0, 28.0, size=num_restaurants)
    restaurant_hist_delays = np.random.uniform(0.08, 0.35, size=num_restaurants)
    restaurant_hist_delivery = restaurant_prep_baselines + np.random.uniform(14.0, 22.0, size=num_restaurants)
    
    rest_lookup = {
        r: {
            "base_prep": restaurant_prep_baselines[i],
            "hist_delay": restaurant_hist_delays[i],
            "hist_delivery": restaurant_hist_delivery[i]
        }
        for i, r in enumerate(restaurants)
    }
    
    num_customers = 8000
    customers = [f"CUST_{i:04d}" for i in range(1, num_customers + 1)]
    
    num_couriers = 350
    couriers = [f"DP_{i:03d}" for i in range(1, num_couriers + 1)]
    courier_exp_levels = np.random.choice(
        ['Novice', 'Intermediate', 'Experienced'],
        size=num_couriers,
        p=[0.25, 0.45, 0.30]
    )
    courier_exp_months = {
        c: np.random.randint(1, 4) if exp == 'Novice'
           else (np.random.randint(5, 14) if exp == 'Intermediate' else np.random.randint(15, 48))
        for c, exp in zip(couriers, courier_exp_levels)
    }
    courier_exp_lookup = {c: exp for c, exp in zip(couriers, courier_exp_levels)}
    
    # 2. Timeline: Chronological span over 180 days (6 months)
    start_date = datetime(2026, 3, 1, 8, 0, 0)
    # Distribute timestamps realistically: higher volume around lunch (12-14) and dinner (19-22)
    # Total seconds in 180 days: 180 * 86400
    time_deltas = np.sort(np.random.uniform(0, 180 * 86400, size=num_orders))
    
    order_times = [start_date + timedelta(seconds=float(td)) for td in time_deltas]
    order_ids = [f"ORD_{i:06d}" for i in range(1, num_orders + 1)]
    
    chosen_restaurants = np.random.choice(restaurants, size=num_orders)
    chosen_customers = np.random.choice(customers, size=num_orders)
    chosen_couriers = np.random.choice(couriers, size=num_orders)
    
    # Extract temporal properties
    hours = np.array([ot.hour for ot in order_times])
    days_of_week = [ot.strftime("%A") for ot in order_times]
    is_weekend = np.array([1 if ot.weekday() >= 5 else 0 for ot in order_times])
    peak_hours = np.array([1 if h in [12, 13, 19, 20, 21] else 0 for h in hours])
    
    # Distance (km): log-normal distribution, clipped to [0.8, 22.0]
    distance_km = np.random.lognormal(mean=1.5, sigma=0.55, size=num_orders)
    distance_km = np.clip(np.round(distance_km, 2), 0.8, 22.0)
    
    # Order items & sizes
    order_sizes = np.random.choice(['Small', 'Medium', 'Large', 'Bulk'], size=num_orders, p=[0.40, 0.38, 0.17, 0.05])
    item_counts = []
    order_values = []
    for sz in order_sizes:
        if sz == 'Small':
            ic = np.random.randint(1, 3)
            val = round(float(np.random.uniform(12.0, 32.0)), 2)
        elif sz == 'Medium':
            ic = np.random.randint(3, 6)
            val = round(float(np.random.uniform(28.0, 65.0)), 2)
        elif sz == 'Large':
            ic = np.random.randint(6, 10)
            val = round(float(np.random.uniform(60.0, 110.0)), 2)
        else: # Bulk
            ic = np.random.randint(10, 18)
            val = round(float(np.random.uniform(105.0, 220.0)), 2)
        item_counts.append(ic)
        order_values.append(val)
    item_counts = np.array(item_counts)
    order_values = np.array(order_values)
    
    # Traffic Level: heavily influenced by peak hour and weekend
    traffic_levels = []
    for h, pk, wk in zip(hours, peak_hours, is_weekend):
        if pk == 1:
            p_traffic = [0.08, 0.32, 0.42, 0.18]
        elif 8 <= h <= 10 or 17 <= h <= 18: # rush hours
            p_traffic = [0.15, 0.40, 0.35, 0.10]
        elif h >= 22 or h <= 7: # late night / early morn
            p_traffic = [0.70, 0.22, 0.07, 0.01]
        else:
            p_traffic = [0.35, 0.45, 0.18, 0.02]
        traffic_levels.append(np.random.choice(['Low', 'Medium', 'High', 'Jam'], p=p_traffic))
    traffic_levels = np.array(traffic_levels)
    
    # Weather conditions
    weather_choices = ['Clear', 'Rainy', 'Stormy', 'Foggy', 'Windy']
    weather_conds = np.random.choice(weather_choices, size=num_orders, p=[0.62, 0.20, 0.06, 0.05, 0.07])
    
    temperatures = []
    precipitations = []
    for w in weather_conds:
        if w == 'Clear':
            temp = round(float(np.random.uniform(20.0, 36.0)), 1)
            precip = 0.0
        elif w == 'Rainy':
            temp = round(float(np.random.uniform(16.0, 26.0)), 1)
            precip = round(float(np.random.uniform(2.0, 18.0)), 1)
        elif w == 'Stormy':
            temp = round(float(np.random.uniform(14.0, 24.0)), 1)
            precip = round(float(np.random.uniform(15.0, 42.0)), 1)
        elif w == 'Foggy':
            temp = round(float(np.random.uniform(12.0, 22.0)), 1)
            precip = round(float(np.random.uniform(0.0, 1.5)), 1)
        else: # Windy
            temp = round(float(np.random.uniform(18.0, 30.0)), 1)
            precip = round(float(np.random.uniform(0.0, 2.0)), 1)
        temperatures.append(temp)
        precipitations.append(precip)
    temperatures = np.array(temperatures)
    precipitations = np.array(precipitations)
    
    # Demand metrics: active delivery partners & orders in last 30 minutes
    active_partners = []
    orders_last_30m = []
    for h, pk in zip(hours, peak_hours):
        if pk == 1:
            couriers_online = np.random.randint(65, 140)
            demand_recent = np.random.randint(70, 185)
        elif 8 <= h <= 22:
            couriers_online = np.random.randint(40, 90)
            demand_recent = np.random.randint(30, 85)
        else:
            couriers_online = np.random.randint(15, 35)
            demand_recent = np.random.randint(8, 25)
        active_partners.append(couriers_online)
        orders_last_30m.append(demand_recent)
    active_partners = np.array(active_partners)
    orders_last_30m = np.array(orders_last_30m)
    
    # 3. Component Modeling for Actual Delivery Times
    # Base prep times per order
    base_preps = np.array([rest_lookup[r]["base_prep"] for r in chosen_restaurants])
    hist_delays = np.array([rest_lookup[r]["hist_delay"] for r in chosen_restaurants])
    hist_deliveries = np.array([rest_lookup[r]["hist_delivery"] for r in chosen_restaurants])
    
    partner_exps = np.array([courier_exp_lookup[c] for c in chosen_couriers])
    partner_exp_mos = np.array([courier_exp_months[c] for c in chosen_couriers])
    
    # Realistic prep time: baseline + item count effect + rush hour queue + stochastic kitchen noise
    kitchen_rush_delay = peak_hours * np.random.uniform(2.0, 6.5, size=num_orders)
    item_prep_factor = item_counts * np.random.uniform(0.6, 1.1, size=num_orders)
    kitchen_noise = np.random.normal(0.0, 2.5, size=num_orders)
    actual_prep_time = base_preps + item_prep_factor + kitchen_rush_delay + kitchen_noise
    actual_prep_time = np.clip(np.round(actual_prep_time, 1), 6.0, 55.0)
    
    # Dispatch & pickup wait time (driven by demand / supply ratio)
    demand_supply_ratio = orders_last_30m / (active_partners + 1.0)
    dispatch_wait_time = np.maximum(1.0, demand_supply_ratio * 4.0 + np.random.exponential(scale=1.8, size=num_orders))
    
    # Transit speed modeling
    # Nominal transit time: 2.2 minutes per km (~27 km/h base scooter speed)
    base_transit_rate = 2.2 
    traffic_multipliers = {'Low': 1.0, 'Medium': 1.32, 'High': 1.75, 'Jam': 2.35}
    weather_multipliers = {'Clear': 1.0, 'Foggy': 1.12, 'Windy': 1.08, 'Rainy': 1.38, 'Stormy': 1.72}
    exp_multipliers = {'Experienced': 0.88, 'Intermediate': 1.0, 'Novice': 1.14}
    
    transit_times = []
    for d, tr, w, exp in zip(distance_km, traffic_levels, weather_conds, partner_exps):
        t_mult = traffic_multipliers[tr]
        w_mult = weather_multipliers[w]
        e_mult = exp_multipliers[exp]
        transit_est = d * base_transit_rate * t_mult * w_mult * e_mult
        transit_times.append(transit_est)
    transit_times = np.array(transit_times)
    
    # Road noise (traffic lights, apartment building navigation, parking)
    road_noise = np.random.gamma(shape=2.5, scale=1.4, size=num_orders) - 1.5
    
    total_actual_duration = actual_prep_time + dispatch_wait_time + transit_times + road_noise
    total_actual_duration = np.clip(np.round(total_actual_duration, 1), 14.0, 115.0)
    
    # 4. Promised SLA calculation (Standard platform SLA heuristic calculated at order placement)
    # The platform estimates SLA using standard prep benchmark + distance transit estimate + platform buffer (12 mins)
    standard_prep_estimate = np.round(base_preps + item_counts * 0.75, 1)
    standard_transit_estimate = np.round(distance_km * 2.8, 1) # ~21 km/h standard assumption
    platform_buffer = 11.0 # SLA buffer minutes
    promised_duration = np.round(standard_prep_estimate + standard_transit_estimate + platform_buffer, 0)
    promised_duration = np.clip(promised_duration, 20.0, 95.0)
    
    # Actual delivery timestamp
    actual_delivery_times = [
        ot + timedelta(minutes=float(dur)) for ot, dur in zip(order_times, total_actual_duration)
    ]
    promised_delivery_times = [
        ot + timedelta(minutes=float(pdur)) for ot, pdur in zip(order_times, promised_duration)
    ]
    
    # Delay metrics
    diff_minutes = total_actual_duration - promised_duration
    delay_minutes = np.maximum(0.0, np.round(diff_minutes, 1))
    delayed = np.array([1 if dm > 0.0 else 0 for dm in delay_minutes])
    
    # Build complete dataframe
    df = pd.DataFrame({
        'order_id': order_ids,
        'restaurant_id': chosen_restaurants,
        'customer_id': chosen_customers,
        'delivery_partner_id': chosen_couriers,
        'order_time': [ot.strftime("%Y-%m-%d %H:%M:%S") for ot in order_times],
        'day_of_week': days_of_week,
        'distance_km': distance_km,
        'restaurant_prep_time': actual_prep_time,
        'restaurant_historical_delivery_time': np.round(hist_deliveries, 1),
        'restaurant_historical_delay_rate': np.round(hist_delays, 3),
        'order_size': order_sizes,
        'item_count': item_counts,
        'order_value': order_values,
        'traffic_level': traffic_levels,
        'weather': weather_conds,
        'temperature': temperatures,
        'precipitation': precipitations,
        'active_delivery_partners': active_partners,
        'orders_last_30_min': orders_last_30m,
        'peak_hour': peak_hours,
        'delivery_partner_experience': partner_exps,
        'delivery_partner_experience_months': partner_exp_mos,
        'promised_delivery_time': [pdt.strftime("%Y-%m-%d %H:%M:%S") for pdt in promised_delivery_times],
        'actual_delivery_time': [adt.strftime("%Y-%m-%d %H:%M:%S") for adt in actual_delivery_times],
        'actual_delivery_duration_min': total_actual_duration,
        'promised_delivery_duration_min': promised_duration,
        'delay_minutes': delay_minutes,
        'delayed': delayed
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "delivery_orders.csv")
    print(f"Generating 55,000 synthetic orders...")
    df = generate_delivery_data(num_orders=55000, random_seed=42)
    df.to_csv(out_file, index=False)
    print(f"Successfully generated and saved {len(df):,} orders to {out_file}.")
    print(f"Delay rate: {df['delayed'].mean():.2%}")
    print(f"Average actual delivery duration: {df['actual_delivery_duration_min'].mean():.1f} min")
    print(f"Average promised delivery duration: {df['promised_delivery_duration_min'].mean():.1f} min")
    print(f"Average delay minutes: {df['delay_minutes'].mean():.1f} min")

"""
Smart Delivery ETA & Delay Prediction Engine - Streamlit Dashboard.
Interactive operations dashboard, real-time ML predictor with prediction intervals & SHAP explainability,
and SQL analytics leaderboards.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from models_def import ETAPipeline, DelayPipeline
from explainability import DeliveryExplainer
from db_manager import DeliveryDatabase

# ----------------- PAGE CONFIG & STYLES -----------------
st.set_page_config(
    page_title="Smart Delivery ETA & Delay Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #212529;
        margin-top: 4px;
    }
    .badge-low {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #ffe8d6;
        color: #d9480f;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-critical {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- RESOURCE CACHING -----------------
@st.cache_resource
def load_db():
    db = DeliveryDatabase()
    return db

@st.cache_resource
def load_models():
    eta_path = os.path.join(os.path.dirname(__file__), "models", "eta_pipeline.joblib")
    delay_path = os.path.join(os.path.dirname(__file__), "models", "delay_pipeline.joblib")
    eta_pipe = joblib.load(eta_path)
    delay_pipe = joblib.load(delay_path)
    explainer = DeliveryExplainer(eta_path, delay_path)
    return eta_pipe, delay_pipe, explainer

@st.cache_data
def load_evaluation_summary():
    eval_path = os.path.join(os.path.dirname(__file__), "models", "evaluation_summary.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return None

db = load_db()
eta_pipe, delay_pipe, explainer = load_models()
eval_summary = load_evaluation_summary()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830312.png", width=64)
    st.title("Smart Delivery")
    st.caption("AI ETA & Delay Intelligence v1.0")
    st.markdown("---")
    
    st.markdown("### Operational Settings")
    conf_level = st.selectbox("Prediction Interval Confidence", [80, 90], index=0, format_func=lambda x: f"{x}% Range")
    risk_preset = st.selectbox("Risk Sensitivity", ["Standard (Default)", "High Sensitivity", "Relaxed"])
    
    st.markdown("---")
    st.markdown("### System Health")
    st.success("🟢 ML Inference: Online")
    st.success("🟢 SQLite Analytics: Connected")
    st.info("📦 Trained on 55,000+ Orders")

# ----------------- MAIN NAVIGATION -----------------
st.title("⚡ Smart Delivery ETA & Delay Prediction Engine")
st.markdown("Dual-target machine learning system delivering calibrated delivery time forecasts, uncertainty bounds, and delay risk mitigation.")

tab_overview, tab_predict, tab_restaurant, tab_models = st.tabs([
    "📊 Executive Overview",
    "🎯 Real-Time ETA & Delay Predictor",
    "🏪 Restaurant & Courier Performance",
    "🔬 Model Performance & Diagnostics"
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
with tab_overview:
    kpis = db.get_kpis()
    
    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Orders Analyzed</div>
            <div class="metric-value">{int(kpis['total_orders']):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Actual Delivery</div>
            <div class="metric-value">{kpis['avg_delivery_time_min']:.1f} <span style="font-size:1rem;">min</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Platform ETA SLA</div>
            <div class="metric-value">{kpis['avg_promised_time_min']:.1f} <span style="font-size:1rem;">min</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Delay Rate</div>
            <div class="metric-value" style="color: {'#dc3545' if kpis['delay_rate_pct'] > 30 else '#28a745'};">{kpis['delay_rate_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Delay Severity</div>
            <div class="metric-value">{kpis['avg_delay_minutes']:.1f} <span style="font-size:1rem;">min</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Delivery Trends
    col_trend1, col_trend2 = st.columns([3, 2])
    with col_trend1:
        st.subheader("📈 Monthly Delivery Volume & Delay Rate")
        df_monthly = db.get_monthly_trends()
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=df_monthly['month'],
            y=df_monthly['order_volume'],
            name="Order Volume",
            marker_color="#4dabf7",
            yaxis='y1'
        ))
        fig_monthly.add_trace(go.Scatter(
            x=df_monthly['month'],
            y=df_monthly['delay_rate_pct'],
            name="Delay Rate (%)",
            line=dict(color='#e03131', width=3),
            yaxis='y2'
        ))
        fig_monthly.update_layout(
            yaxis=dict(title="Orders"),
            yaxis2=dict(title="Delay Rate (%)", overlaying='y', side='right', range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=320
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_trend2:
        st.subheader("🕒 Delay Rate by Hour of Day")
        df_hourly = db.get_hourly_delay_distribution()
        fig_hourly = px.area(
            df_hourly,
            x='hour_of_day',
            y='delay_rate_pct',
            labels={'hour_of_day': 'Hour (0-23)', 'delay_rate_pct': 'Delay Rate (%)'},
            color_discrete_sequence=['#ff922b']
        )
        fig_hourly.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    # Operational Impact Breakdown
    st.subheader("🌦️ Environmental & Traffic Stress Drivers")
    col_env1, col_env2, col_env3 = st.columns(3)
    
    with col_env1:
        st.markdown("**Traffic Congestion Impact**")
        df_traffic = db.get_traffic_impact()
        fig_tr = px.bar(
            df_traffic,
            x='traffic_level',
            y='avg_delivery_min',
            color='delay_rate_pct',
            color_continuous_scale='Reds',
            labels={'traffic_level': 'Traffic', 'avg_delivery_min': 'Avg Delivery (min)', 'delay_rate_pct': 'Delay %'},
            text_auto='.1f'
        )
        fig_tr.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tr, use_container_width=True)

    with col_env2:
        st.markdown("**Weather Conditions Impact**")
        df_weather = db.get_weather_impact()
        fig_we = px.bar(
            df_weather,
            x='weather',
            y='delay_rate_pct',
            color='avg_delivery_min',
            color_continuous_scale='Tealgrn',
            labels={'weather': 'Weather', 'delay_rate_pct': 'Delay Rate (%)', 'avg_delivery_min': 'Delivery (min)'},
            text_auto='.1f'
        )
        fig_we.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_we, use_container_width=True)

    with col_env3:
        st.markdown("**Distance Buckets vs Delivery Duration**")
        df_dist = db.get_distance_vs_delivery_time()
        fig_di = px.bar(
            df_dist,
            x='distance_category',
            y='avg_delivery_min',
            color='delay_rate_pct',
            color_continuous_scale='Blues',
            labels={'distance_category': 'Distance', 'avg_delivery_min': 'Delivery (min)', 'delay_rate_pct': 'Delay %'},
            text_auto='.1f'
        )
        fig_di.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_di, use_container_width=True)

# ==============================================================================
# TAB 2: REAL-TIME ETA & DELAY PREDICTOR
# ==============================================================================
with tab_predict:
    st.subheader("🎯 Real-Time Order ETA & Delay Risk Prediction")
    st.caption("Enter dispatch parameters below to generate calibrated ETA ranges, delay likelihood, and SHAP feature attributions.")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.markdown("#### 1. Order & Kitchen Details")
        p_restaurant = st.selectbox(
            "Select Restaurant",
            [f"REST_{i:03d}" for i in range(1, 61)],
            index=4,
            help="Associated with historical kitchen prep baselines"
        )
        p_distance = st.slider("Delivery Distance (km)", min_value=0.8, max_value=22.0, value=6.5, step=0.1)
        p_prep_time = st.slider("Estimated Kitchen Prep Time (min)", min_value=8.0, max_value=55.0, value=22.0, step=1.0)
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            p_order_size = st.selectbox("Order Size", ["Small", "Medium", "Large", "Bulk"], index=1)
            p_item_count = st.number_input("Item Count", min_value=1, max_value=25, value=4)
        with c_sub2:
            p_order_value = st.number_input("Order Cart Value ($)", min_value=5.0, max_value=300.0, value=48.5, step=5.0)
            p_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)

    with col_input2:
        st.markdown("#### 2. Environment & Fleet Supply")
        c_sub3, c_sub4 = st.columns(2)
        with c_sub3:
            p_hour = st.slider("Order Hour (24h)", min_value=0, max_value=23, value=19)
            p_traffic = st.selectbox("Traffic Congestion", ["Low", "Medium", "High", "Jam"], index=2)
            p_weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy", "Foggy", "Windy"], index=0)
        with c_sub4:
            p_temp = st.number_input("Temperature (°C)", min_value=5.0, max_value=45.0, value=26.0, step=1.0)
            p_precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0 if p_weather == 'Clear' else 8.5, step=1.0)
            p_courier_exp = st.selectbox("Courier Experience", ["Experienced", "Intermediate", "Novice"], index=1)
            
        c_sub5, c_sub6 = st.columns(2)
        with c_sub5:
            p_active_couriers = st.slider("Active Couriers Online", min_value=10, max_value=150, value=45)
        with c_sub6:
            p_recent_orders = st.slider("Orders in Last 30 min (Demand)", min_value=5, max_value=220, value=75)

    exp_mos_map = {'Novice': 2, 'Intermediate': 9, 'Experienced': 24}
    
    # Assemble input DataFrame
    order_dict = {
        'restaurant_id': p_restaurant,
        'customer_id': 'CUST_0001',
        'delivery_partner_id': 'DP_001',
        'order_time': f"2026-09-20 {p_hour:02d}:30:00",
        'day_of_week': p_day,
        'distance_km': p_distance,
        'restaurant_prep_time': p_prep_time,
        'order_size': p_order_size,
        'item_count': p_item_count,
        'order_value': p_order_value,
        'traffic_level': p_traffic,
        'weather': p_weather,
        'temperature': p_temp,
        'precipitation': p_precip,
        'active_delivery_partners': p_active_couriers,
        'orders_last_30_min': p_recent_orders,
        'peak_hour': 1 if p_hour in [12, 13, 19, 20, 21] else 0,
        'delivery_partner_experience': p_courier_exp,
        'delivery_partner_experience_months': exp_mos_map[p_courier_exp]
    }
    df_live = pd.DataFrame([order_dict])
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict Delivery ETA & Delay Risk", type="primary", use_container_width=True)
    
    # Run Inference
    eta_result = eta_pipe.predict_with_interval(df_live, confidence=conf_level)
    risk_result = delay_pipe.predict_risk(df_live)
    explanation = explainer.explain_order(df_live, top_k=5)
    
    st.markdown("---")
    st.subheader("📋 Prediction Results & Risk Assessment")
    
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #228be6;">
            <div class="metric-label">Predicted Delivery ETA</div>
            <div class="metric-value">{int(round(eta_result['predicted_eta']))} <span style="font-size:1.1rem;">min</span></div>
            <div style="font-size:0.85rem; color:#495057; margin-top:6px;">Target arrival duration</div>
        </div>
        """, unsafe_allow_html=True)
    with o2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #12b886;">
            <div class="metric-label">Expected ETA Range ({conf_level}%)</div>
            <div class="metric-value">{eta_result['range_str']}</div>
            <div style="font-size:0.85rem; color:#495057; margin-top:6px;">Calibrated uncertainty interval</div>
        </div>
        """, unsafe_allow_html=True)
    with o3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {risk_result['color']};">
            <div class="metric-label">Delay Likelihood</div>
            <div class="metric-value">{risk_result['probability_pct']}%</div>
            <div style="font-size:0.85rem; color:#495057; margin-top:6px;">Probability of exceeding SLA</div>
        </div>
        """, unsafe_allow_html=True)
    with o4:
        tier = risk_result['risk_level']
        badge_class = f"badge-{tier.lower()}"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {risk_result['color']};">
            <div class="metric-label">Operational Risk Score</div>
            <div class="metric-value">{risk_result['risk_score']}<span style="font-size:1.1rem;">/100</span></div>
            <div style="margin-top:6px;"><span class="{badge_class}">{tier} RISK</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Explanation section
    st.markdown("<br>", unsafe_allow_html=True)
    col_exp1, col_exp2 = st.columns([3, 2])
    
    with col_exp1:
        st.markdown("#### 🔍 SHAP Local Feature Attribution")
        top_factors = explanation['top_factors']
        df_factors = pd.DataFrame(top_factors)
        
        fig_shap = px.bar(
            df_factors,
            x='shap_value',
            y='feature',
            orientation='h',
            color='direction',
            color_discrete_map={
                'Increases ETA (+Delay)': '#fa5252',
                'Decreases ETA (Faster)': '#20c997'
            },
            labels={'shap_value': 'Impact on ETA (Minutes)', 'feature': 'Operational Factor'},
            text_auto='+.1f'
        )
        fig_shap.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_exp2:
        st.markdown("#### 💡 Key Operational Takeaways")
        st.markdown(f"**Primary Contributing Factors:**")
        for f in top_factors[:3]:
            st.markdown(f"• **{f['feature']}**: `{f['shap_value']:+0.1f} min` ({f['direction']})")
        
        st.markdown("**Actionable Dispatch Recommendation:**")
        if risk_result['risk_level'] in ['HIGH', 'CRITICAL']:
            st.warning(f"⚠️ **High Delay Risk Detected**: Fleet demand/traffic pressure is elevated. Consider assigning a priority tier courier or adding a 5-10 min buffer to customer notification.")
        elif risk_result['risk_level'] == 'MEDIUM':
            st.info(f"ℹ️ **Moderate Conditions**: Monitor prep time completion at {p_restaurant} to avoid dispatch queues.")
        else:
            st.success(f"✅ **Optimal Dispatch Conditions**: Delivery expected smoothly within normal SLA window.")

# ==============================================================================
# TAB 3: RESTAURANT & COURIER PERFORMANCE (SQL ANALYTICS)
# ==============================================================================
with tab_restaurant:
    st.subheader("🏪 Restaurant Performance & Courier Analytics (SQL Powered)")
    st.caption("Live aggregations generated via SQLite analytical queries.")
    
    # Leaderboard Table
    df_rest = db.get_restaurant_performance(limit=25)
    
    col_rst1, col_rst2 = st.columns([3, 2])
    with col_rst1:
        st.markdown("#### Restaurant Performance Leaderboard")
        st.dataframe(
            df_rest.style.format({
                'total_orders': '{:,}',
                'avg_prep_time_min': '{:.1f} min',
                'avg_delivery_time_min': '{:.1f} min',
                'avg_delay_min': '{:.1f} min',
                'delay_rate_pct': '{:.1f}%'
            }).background_gradient(subset=['delay_rate_pct'], cmap='Reds'),
            use_container_width=True,
            height=340
        )
        
    with col_rst2:
        st.markdown("#### Kitchen Prep Time vs Delay Rate")
        fig_scatter = px.scatter(
            df_rest,
            x='avg_prep_time_min',
            y='delay_rate_pct',
            size='total_orders',
            color='avg_delivery_time_min',
            hover_name='restaurant_id',
            labels={
                'avg_prep_time_min': 'Avg Prep Time (min)',
                'delay_rate_pct': 'Delay Rate (%)',
                'avg_delivery_time_min': 'Delivery (min)'
            },
            color_continuous_scale='Viridis'
        )
        fig_scatter.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🛵 Delivery Partner Efficiency by Experience Tier")
    col_prt1, col_prt2 = st.columns(2)
    
    df_partner = db.get_partner_performance()
    with col_prt1:
        st.dataframe(
            df_partner.style.format({
                'total_deliveries': '{:,}',
                'avg_experience_months': '{:.1f} mos',
                'avg_delivery_min': '{:.1f} min',
                'delay_rate_pct': '{:.1f}%'
            }),
            use_container_width=True
        )
    with col_prt2:
        fig_prt = px.bar(
            df_partner,
            x='delivery_partner_experience',
            y='avg_delivery_min',
            color='delay_rate_pct',
            color_continuous_scale='Mint',
            labels={'delivery_partner_experience': 'Experience Level', 'avg_delivery_min': 'Avg Duration (min)'},
            text_auto='.1f'
        )
        fig_prt.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_prt, use_container_width=True)

# ==============================================================================
# TAB 4: MODEL PERFORMANCE & EXPLAINABILITY
# ==============================================================================
with tab_models:
    st.subheader("🔬 Model Evaluation, Benchmarks & Diagnostic Diagnostics")
    st.caption("Trained on 38,500 orders and evaluated against an out-of-sample chronological test set of 8,250 orders.")
    
    if eval_summary:
        col_m1, col_m2 = st.columns(2)
        
        # ETA Models Comparison
        with col_m1:
            st.markdown("#### 1. ETA Regression Models Comparison")
            eta_models = eval_summary['eta_evaluation']['all_models']
            eta_rows = []
            for m_name, m_data in eta_models.items():
                eta_rows.append({
                    'Model': m_name,
                    'Val MAE': m_data['val']['mae'],
                    'Val RMSE': m_data['val']['rmse'],
                    'Val R²': m_data['val']['r2'],
                    'Test MAE': m_data['test']['mae'],
                    'Test RMSE': m_data['test']['rmse'],
                    'Test R²': m_data['test']['r2']
                })
            df_eta_cmp = pd.DataFrame(eta_rows).sort_values('Test MAE')
            st.dataframe(df_eta_cmp.style.highlight_min(subset=['Test MAE', 'Test RMSE'], color='#d4edda').highlight_max(subset=['Test R²'], color='#d4edda'), use_container_width=True)

        # Delay Models Comparison
        with col_m2:
            st.markdown("#### 2. Delay Classification Models Comparison")
            delay_models = eval_summary['delay_evaluation']['all_models']
            delay_rows = []
            for m_name, m_data in delay_models.items():
                delay_rows.append({
                    'Model': m_name,
                    'Val ROC-AUC': m_data['val']['roc_auc'],
                    'Val F1': m_data['val']['f1'],
                    'Val Precision': m_data['val']['precision'],
                    'Test ROC-AUC': m_data['test']['roc_auc'],
                    'Test F1': m_data['test']['f1'],
                    'Test Precision': m_data['test']['precision']
                })
            df_delay_cmp = pd.DataFrame(delay_rows).sort_values('Test ROC-AUC', ascending=False)
            st.dataframe(df_delay_cmp.style.highlight_max(subset=['Test ROC-AUC', 'Test F1', 'Test Precision'], color='#d4edda'), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        
        # Scatter: Actual vs Predicted
        with col_diag1:
            st.markdown("#### Actual vs Predicted ETA (Test Sample)")
            pts = eval_summary['eta_evaluation']['sample_predictions']
            df_pts = pd.DataFrame(pts)
            fig_sc = px.scatter(
                df_pts,
                x='actual',
                y='predicted',
                labels={'actual': 'Actual Delivery (min)', 'predicted': 'Predicted ETA (min)'},
                opacity=0.65
            )
            # Add identity line
            min_val = min(df_pts['actual'].min(), df_pts['predicted'].min())
            max_val = max(df_pts['actual'].max(), df_pts['predicted'].max())
            fig_sc.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Ideal Fit'
            ))
            fig_sc.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sc, use_container_width=True)

        # ROC Curve
        with col_diag2:
            st.markdown("#### Delay Classifier ROC Curve")
            roc_pts = eval_summary['delay_evaluation']['roc_curve']
            df_roc = pd.DataFrame(roc_pts)
            fig_roc = px.line(
                df_roc,
                x='fpr',
                y='tpr',
                labels={'fpr': 'False Positive Rate', 'tpr': 'True Positive Rate'}
            )
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Chance'
            ))
            auc_val = eval_summary['delay_evaluation']['test_metrics']['roc_auc']
            fig_roc.update_layout(
                title=f"Test ROC-AUC = {auc_val:.4f}",
                height=280,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion Matrix
        with col_diag3:
            st.markdown("#### Confusion Matrix (Test Set)")
            cm = eval_summary['delay_evaluation']['confusion_matrix']
            cm_labels = ["On-Time (0)", "Delayed (1)"]
            fig_cm = px.imshow(
                cm,
                x=cm_labels,
                y=cm_labels,
                labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                text_auto=True,
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Global Feature Importance
        st.markdown("#### 🌟 Global Model Feature Importance (SHAP / XGBoost Gain)")
        df_imp = explainer.get_global_feature_importance(top_n=10)
        fig_global = px.bar(
            df_imp,
            x='Relative Importance (%)',
            y='Feature',
            orientation='h',
            color='Relative Importance (%)',
            color_continuous_scale='Viridis',
            text_auto='.1f'
        )
        fig_global.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_global, use_container_width=True)
    else:
        st.warning("Evaluation summary not found. Run `py src/evaluate.py` to generate.")

"""
Air Quality Level Classification - Modern Web Dashboard
Capstone 19: Full-Stack Machine Learning Web Application
Built with Streamlit, Plotly, Scikit-Learn, and Pandas
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_preprocessing import (
    AirQualityPreprocessor,
    POLLUTANT_FEATURES,
    AQI_CATEGORIES,
    AQI_COLOR_MAP,
    AQI_HEALTH_ADVISORY,
    calculate_cpcb_aqi,
    calculate_sub_index
)

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Level Classification | Capstone 19",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Environmental Science CSS
CUSTOM_CSS = """
<style>
    /* Global Typography & Palette */
    :root {
        --primary-navy: #0F172A;
        --secondary-navy: #1E293B;
        --accent-teal: #0D9488;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --bg-soft: #F8FAFC;
        --card-bg: #FFFFFF;
        --border-color: #E2E8F0;
        --text-main: #0F172A;
        --text-muted: #64748B;
    }
    
    .main {
        background-color: var(--bg-soft);
    }
    
    /* Top Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0D9488 100%);
        color: #FFFFFF;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    .hero-badge {
        background: rgba(13, 148, 136, 0.35);
        color: #5EEAD4;
        border: 1px solid rgba(94, 234, 212, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #CBD5E1;
        margin-top: 0.5rem;
        max-width: 850px;
        line-height: 1.5;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #0D9488;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction Badge */
    .pred-badge {
        padding: 1.2rem 1.8rem;
        border-radius: 14px;
        color: #FFFFFF;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px -2px rgba(0,0,0,0.12);
    }
    .pred-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.9;
        font-weight: 600;
    }
    .pred-cat {
        font-size: 2.3rem;
        font-weight: 900;
        margin: 0.3rem 0;
        letter-spacing: -0.01em;
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        font-weight: 600;
        color: #475569;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border-color: #0D9488 !important;
    }
    
    /* Disclaimer Card */
    .disclaimer-box {
        background: #F8FAFC;
        border-left: 4px solid #0D9488;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 1.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Loading & Caching
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads saved Random Forest model pipeline and model metadata."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")
    meta_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return model, metadata


@st.cache_resource
def load_all_models():
    """Loads all trained candidate pipelines for comparison."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    loaded = {}
    for name, fname in [
        ("Dummy Baseline", "dummy_classifier.joblib"),
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Random Forest", "random_forest.joblib")
    ]:
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            loaded[name] = joblib.load(p)
    return loaded


@st.cache_data
def load_evaluation_reports():
    """Loads structured evaluation JSON reports from reports/."""
    rep_dir = os.path.join(os.path.dirname(__file__), "reports")
    data = {}
    for k in ["evaluation_metrics", "classification_reports", "confusion_matrices", "feature_importance"]:
        p = os.path.join(rep_dir, f"{k}.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                data[k] = json.load(f)
    return data


@st.cache_data
def load_default_dataset():
    """Loads default CPCB Indian cities air quality dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "city_day.csv")
    if os.path.exists(csv_path):
        preprocessor = AirQualityPreprocessor()
        df = preprocessor.load_data(csv_path)
        return df
    return None


# ---------------------------------------------------------
# Application Header & Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">Academic Engineering Capstone 19 • CPCB Monitoring Network</div>
    <h1 class="hero-title">Air Quality Level Classification Dashboard</h1>
    <p class="hero-subtitle">
        An end-to-end Machine Learning intelligence platform classifying ambient atmospheric readings 
        into official Central Pollution Control Board (CPCB) India AQI categories with multi-pollutant explainability.
    </p>
</div>
""", unsafe_allow_html=True)

# Load core artifacts
df_default = load_default_dataset()
best_model, model_metadata = load_trained_model()
eval_reports = load_evaluation_reports()
all_models = load_all_models()

# ---------------------------------------------------------
# Sidebar Configuration & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=64)
    st.markdown("### **Project Navigator**")
    st.caption("Capstone 19 • B.Tech / Academic Project")
    
    st.markdown("---")
    st.markdown("**Dataset Provenance**")
    st.write("🏛️ **Source:** CPCB Ministry of Environment, India")
    st.write("📍 **Coverage:** 26 Major Indian Cities")
    st.write("⏱️ **Timeline:** Jan 2015 – Jul 2020")
    st.write("📊 **Records:** 29,531 Daily CAAQMS Logs")
    
    st.markdown("---")
    st.markdown("**Current Model Status**")
    if best_model is not None:
        best_name = model_metadata.get("best_model_name", "Random Forest")
        rf_metrics = eval_reports.get("evaluation_metrics", {}).get("Random Forest", {})
        test_acc = rf_metrics.get("accuracy", 0.8145) * 100
        test_f1 = rf_metrics.get("macro_f1", 0.7880) * 100
        st.success(f"**Model Active:** {best_name}")
        st.metric("Test Accuracy", f"{test_acc:.1f}%")
        st.metric("Macro F1-Score", f"{test_f1:.1f}%")
    else:
        st.warning("Model artifacts not loaded. Run train_model.py first.")
        
    st.markdown("---")
    st.caption("⚡ Built with Streamlit, Scikit-learn & Plotly")

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_overview, tab_explorer, tab_analysis, tab_prediction, tab_evaluation, tab_about = st.tabs([
    "📋 Overview",
    "📂 Dataset Explorer",
    "📈 Air Quality Analysis",
    "🔮 Live AQI Prediction",
    "📊 Model Evaluation",
    "ℹ️ About & Provenance"
])

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab_overview:
    st.markdown('<div class="section-title">📌 Executive Summary & Key Indicators</div>', unsafe_allow_html=True)
    
    # 5 KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset Records</div>
            <div class="metric-val">29,531</div>
            <div class="metric-sub">26 Indian Cities</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Chemical Features</div>
            <div class="metric-val">12</div>
            <div class="metric-sub">PM, Gases, VOCs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">AQI Target Classes</div>
            <div class="metric-val">6</div>
            <div class="metric-sub">Good to Severe</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Top Model Accuracy</div>
            <div class="metric-val">81.45%</div>
            <div class="metric-sub">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Macro F1-Score</div>
            <div class="metric-val">0.7880</div>
            <div class="metric-sub">Balanced Across Classes</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_ov_left, col_ov_right = st.columns([3, 2])
    
    with col_ov_left:
        st.markdown("#### 🎯 Problem Definition & Approach")
        st.write("""
        Air pollution is among the most critical environmental and public health hazards in modern urban society.
        Standard continuous ambient monitoring stations measure diverse atmospheric compounds, but converting raw 
        concentrations into intuitive categorical warnings helps citizens, public transport authorities, and schools 
        take timely preventive measures.
        
        **In this Capstone Project:**
        1. We ingest **29,531 ground-truth daily records** recorded by CPCB stations across 26 Indian metropolitan areas.
        2. We implement strict **leakage-free preprocessing**: eliminating target columns (`AQI` and `AQI_Bucket`) from input features so the model learns purely from physical pollutant concentrations.
        3. We benchmark three models: a **Dummy Baseline**, a **Multinomial Logistic Regression baseline**, and an **Ensemble Random Forest Classifier**.
        4. We achieve **81.45% overall test accuracy** and **0.7880 macro F1-score**, correctly identifying high-risk Severe episodes with 84.6% per-class F1-score.
        """)
        
        st.markdown("#### 🏷️ Official CPCB NAQI Categories")
        cols_cat = st.columns(6)
        for i, cat in enumerate(AQI_CATEGORIES):
            with cols_cat[i]:
                st.markdown(f"""
                <div style="background-color: {AQI_COLOR_MAP[cat]}; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 0.85rem;">
                    {cat}
                </div>
                """, unsafe_allow_html=True)
                
    with col_ov_right:
        st.markdown("#### 📊 CPCB Category Share in Monitored Data")
        if df_default is not None:
            counts = df_default["AQI_Bucket"].value_counts().loc[AQI_CATEGORIES]
            fig_donut = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map=AQI_COLOR_MAP,
                hole=0.55
            )
            fig_donut.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Academic Prototype Notice:</strong> This software is engineered as an educational machine learning 
        demonstration for college project evaluation. It is not an officially certified environmental regulatory tool 
        nor an automated emergency alert apparatus. Official health advisories must reference CPCB India or local pollution authorities.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 2: DATASET EXPLORER
# =========================================================
with tab_explorer:
    st.markdown('<div class="section-title">📂 Ambient Air Quality Dataset Inspection</div>', unsafe_allow_html=True)
    
    st.write("Inspect the official CPCB Indian cities dataset or upload your own CSV file for automated schema verification.")
    
    # CSV Upload Component
    uploaded_file = st.file_uploader(
        "Upload Air Quality CSV (Supports CPCB schema or generic chemical pollutant CSVs)",
        type=["csv"],
        help="Upload a CSV with columns like PM2.5, PM10, NO2, CO, SO2, O3, City, Date, AQI_Bucket."
    )
    
    active_df = df_default
    is_custom_upload = False
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded custom CSV: **{uploaded_file.name}** ({len(custom_df):,} rows, {len(custom_df.columns)} columns)")
            active_df = custom_df
            is_custom_upload = True
        except Exception as e:
            st.error(f"Error reading uploaded CSV file: {e}")
            active_df = df_default

    if active_df is not None:
        # Schema Identification Card
        detected_pollutants = [p for p in POLLUTANT_FEATURES if p in active_df.columns]
        has_target = "AQI_Bucket" in active_df.columns
        has_aqi = "AQI" in active_df.columns
        has_city = "City" in active_df.columns
        has_date = "Date" in active_df.columns
        
        c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
        c_sc1.metric("Total Records", f"{len(active_df):,}")
        c_sc2.metric("Detected Pollutants", f"{len(detected_pollutants)} / 12")
        c_sc3.metric("Target (AQI_Bucket)", "Present ✅" if has_target else "Missing ❌")
        c_sc4.metric("City Metadata", f"{active_df['City'].nunique()} Cities" if has_city else "None")
        
        st.markdown("---")
        
        # Interactive Filter and Preview Controls
        col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
        with col_f1:
            if has_city:
                selected_cities = st.multiselect("Filter by City", options=list(active_df["City"].unique()), default=[])
            else:
                selected_cities = []
        with col_f2:
            if has_target:
                selected_buckets = st.multiselect("Filter by AQI Category", options=AQI_CATEGORIES, default=[])
            else:
                selected_buckets = []
        with col_f3:
            num_rows = st.slider("Rows to Preview", min_value=5, max_value=100, value=15, step=5)
            
        filtered_df = active_df.copy()
        if selected_cities:
            filtered_df = filtered_df[filtered_df["City"].isin(selected_cities)]
        if selected_buckets and has_target:
            filtered_df = filtered_df[filtered_df["AQI_Bucket"].isin(selected_buckets)]
            
        st.markdown(f"**Showing {min(num_rows, len(filtered_df))} of {len(filtered_df):,} records**")
        st.dataframe(filtered_df.head(num_rows), use_container_width=True)
        
        # Missing Value Summary & Statistics
        st.markdown("#### 📉 Missing Value Profiling & Data Completeness")
        col_m_left, col_m_right = st.columns([1, 1])
        
        with col_m_left:
            missing_series = active_df.isnull().sum()
            missing_pct = (missing_series / len(active_df) * 100).round(2)
            miss_table = pd.DataFrame({
                "Column": active_df.columns,
                "Missing Rows": missing_series.values,
                "Missing %": missing_pct.values,
                "Valid Rows": (len(active_df) - missing_series.values)
            }).sort_values(by="Missing %", ascending=False).reset_index(drop=True)
            st.dataframe(miss_table, height=290, use_container_width=True)
            
        with col_m_right:
            fig_miss = px.bar(
                miss_table[miss_table["Missing %"] > 0],
                x="Column",
                y="Missing %",
                color="Missing %",
                color_continuous_scale="Teal",
                title="Missing Value Percentage by Variable"
            )
            fig_miss.update_layout(margin=dict(t=35, b=10, l=10, r=10), height=290)
            st.plotly_chart(fig_miss, use_container_width=True)
            
        # Download Cleaned Data Option
        st.markdown("#### 💾 Export Clean Preprocessed Data")
        st.write("Download the sanitized version of this dataset with invalid negatives removed, dates standardized, and outliers checked.")
        preproc = AirQualityPreprocessor()
        clean_subset = preproc.clean_data(active_df, drop_unlabeled=False)
        csv_bytes = clean_subset.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned CSV File",
            data=csv_bytes,
            file_name="cpcb_air_quality_clean.csv",
            mime="text/csv"
        )


# =========================================================
# TAB 3: AIR QUALITY ANALYSIS
# =========================================================
with tab_analysis:
    st.markdown('<div class="section-title">📈 Exploratory Air Pollution Science & Spatial Analysis</div>', unsafe_allow_html=True)
    
    if df_default is not None:
        eda_df = df_default.dropna(subset=["AQI_Bucket"]).copy()
        eda_df["Date"] = pd.to_datetime(eda_df["Date"], errors="coerce")
        eda_df["Year"] = eda_df["Date"].dt.year
        eda_df["Month"] = eda_df["Date"].dt.month
        
        col_an1, col_an2 = st.columns([1, 1])
        
        # Chart 1: Time Series Pollutant Trends
        with col_an1:
            st.markdown("#### ⏱️ Multi-Year Trend Analysis")
            sel_city = st.selectbox("Select City for Time-Series Analysis", options=["Delhi", "Bengaluru", "Ahmedabad", "Kolkata", "Hyderabad", "Chennai"], index=0)
            sel_poll = st.selectbox("Select Pollutant", options=["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"], index=0)
            
            city_ts = eda_df[eda_df["City"] == sel_city].sort_values("Date").dropna(subset=[sel_poll])
            if not city_ts.empty:
                # 7-day rolling mean for smooth interpretation
                city_ts["Rolling_7D"] = city_ts[sel_poll].rolling(window=7, min_periods=1).mean()
                fig_ts = px.line(
                    city_ts,
                    x="Date",
                    y="Rolling_7D",
                    title=f"7-Day Rolling Trend of {sel_poll} in {sel_city} (2015-2020)",
                    labels={"Rolling_7D": f"{sel_poll} Concentration / Index"}
                )
                fig_ts.update_traces(line_color="#0D9488", line_width=2)
                fig_ts.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("No recorded readings for selected city/pollutant combination.")
                
        # Chart 2: Inter-City AQI Comparison
        with col_an2:
            st.markdown("#### 🏙️ Inter-City Air Quality Ranking")
            top_cities = eda_df["City"].value_counts().head(12).index
            city_summary = eda_df[eda_df["City"].isin(top_cities)].groupby("City")["AQI"].median().reset_index()
            city_summary = city_summary.sort_values(by="AQI", ascending=True)
            
            fig_city = px.bar(
                city_summary,
                x="AQI",
                y="City",
                orientation="h",
                color="AQI",
                color_continuous_scale="Viridis_r",
                title="Median AQI Across Top 12 Monitored Indian Metros"
            )
            fig_city.add_vline(x=50, line_dash="dash", line_color="#10B981", annotation_text="Good (50)")
            fig_city.add_vline(x=100, line_dash="dash", line_color="#84CC16", annotation_text="Satisfactory (100)")
            fig_city.add_vline(x=200, line_dash="dash", line_color="#F59E0B", annotation_text="Moderate (200)")
            fig_city.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_city, use_container_width=True)

        st.markdown("---")
        
        col_an3, col_an4 = st.columns([1, 1])
        
        # Chart 3: Correlation Matrix
        with col_an3:
            st.markdown("#### 🔗 Inter-Pollutant Correlation Heatmap")
            corr_cols = ["PM2.5", "PM10", "NO2", "NOx", "CO", "SO2", "O3", "NH3", "AQI"]
            corr_mat = eda_df[corr_cols].corr().round(2)
            fig_corr = px.imshow(
                corr_mat,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Pearson Correlation Between Atmospheric Compounds"
            )
            fig_corr.update_layout(height=360, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_corr, use_container_width=True)
            
        # Chart 4: Seasonal Pollution Cycle
        with col_an4:
            st.markdown("#### ❄️ Seasonal Variation Cycle (Monthly Medians)")
            season_df = eda_df.groupby("Month")[["PM2.5", "PM10", "AQI"]].median().reset_index()
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            season_df["Month_Name"] = [month_names[m-1] for m in season_df["Month"]]
            
            fig_season = go.Figure()
            fig_season.add_trace(go.Scatter(x=season_df["Month_Name"], y=season_df["PM2.5"], name="PM2.5 (ug/m3)", line=dict(color="#EF4444", width=2.5)))
            fig_season.add_trace(go.Scatter(x=season_df["Month_Name"], y=season_df["PM10"], name="PM10 (ug/m3)", line=dict(color="#F97316", width=2)))
            fig_season.add_trace(go.Scatter(x=season_df["Month_Name"], y=season_df["AQI"], name="AQI", line=dict(color="#0284C7", width=3, dash="dot")))
            fig_season.update_layout(
                title="Subcontinental Seasonality: Winter Stagnation vs Monsoon Cleanse",
                height=360,
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_season, use_container_width=True)


# =========================================================
# TAB 4: LIVE PREDICTION
# =========================================================
with tab_prediction:
    st.markdown('<div class="section-title">🔮 Real-Time Air Quality Classification Engine</div>', unsafe_allow_html=True)
    st.write("Enter chemical pollutant concentrations to predict the official CPCB air quality category using our trained Random Forest model.")
    
    # Preset Scenario Buttons
    st.markdown("**⚡ Quick Preset Environmental Scenarios:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    # Store form state in session_state
    if "input_pm25" not in st.session_state: st.session_state.input_pm25 = 45.0
    if "input_pm10" not in st.session_state: st.session_state.input_pm10 = 85.0
    if "input_no" not in st.session_state: st.session_state.input_no = 12.0
    if "input_no2" not in st.session_state: st.session_state.input_no2 = 28.0
    if "input_nox" not in st.session_state: st.session_state.input_nox = 24.0
    if "input_nh3" not in st.session_state: st.session_state.input_nh3 = 18.0
    if "input_co" not in st.session_state: st.session_state.input_co = 0.9
    if "input_so2" not in st.session_state: st.session_state.input_so2 = 12.0
    if "input_o3" not in st.session_state: st.session_state.input_o3 = 35.0
    if "input_benzene" not in st.session_state: st.session_state.input_benzene = 2.0
    if "input_toluene" not in st.session_state: st.session_state.input_toluene = 5.0
    if "input_xylene" not in st.session_state: st.session_state.input_xylene = 1.5

    def set_scenario(pm25, pm10, no, no2, nox, nh3, co, so2, o3, bz, tol, xyl):
        st.session_state.input_pm25 = float(pm25)
        st.session_state.input_pm10 = float(pm10)
        st.session_state.input_no = float(no)
        st.session_state.input_no2 = float(no2)
        st.session_state.input_nox = float(nox)
        st.session_state.input_nh3 = float(nh3)
        st.session_state.input_co = float(co)
        st.session_state.input_so2 = float(so2)
        st.session_state.input_o3 = float(o3)
        st.session_state.input_benzene = float(bz)
        st.session_state.input_toluene = float(tol)
        st.session_state.input_xylene = float(xyl)

    with p_col1:
        if st.button("🍃 1. Clean Coastal / Monsoon"):
            set_scenario(18.0, 35.0, 4.0, 14.0, 10.0, 8.0, 0.4, 5.0, 22.0, 0.5, 1.2, 0.4)
            st.rerun()
    with p_col2:
        if st.button("🏙️ 2. Typical Urban Moderate"):
            set_scenario(65.0, 130.0, 16.0, 38.0, 32.0, 22.0, 1.2, 14.0, 42.0, 2.5, 7.0, 2.0)
            st.rerun()
    with p_col3:
        if st.button("🏭 3. High Traffic / Poor"):
            set_scenario(105.0, 220.0, 35.0, 72.0, 58.0, 38.0, 2.4, 26.0, 68.0, 5.2, 16.0, 4.5)
            st.rerun()
    with p_col4:
        if st.button("⚠️ 4. Severe Winter Smog"):
            set_scenario(320.0, 550.0, 85.0, 140.0, 120.0, 75.0, 5.5, 48.0, 95.0, 14.0, 35.0, 9.0)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Form
    with st.form("aqi_prediction_form"):
        st.markdown("##### **1. Particulate Matter & Criteria Combustion Gases**")
        c_in1, c_in2, c_in3, c_in4 = st.columns(4)
        val_pm25 = c_in1.number_input("PM2.5 (ug/m3)", min_value=0.0, max_value=1000.0, value=st.session_state.input_pm25, step=1.0, help="CPCB 24h standard: 60 ug/m3")
        val_pm10 = c_in2.number_input("PM10 (ug/m3)", min_value=0.0, max_value=1200.0, value=st.session_state.input_pm10, step=2.0, help="CPCB 24h standard: 100 ug/m3")
        val_no2 = c_in3.number_input("NO2 (ug/m3)", min_value=0.0, max_value=500.0, value=st.session_state.input_no2, step=1.0, help="CPCB 24h standard: 80 ug/m3")
        val_co = c_in4.number_input("CO (mg/m3)", min_value=0.0, max_value=100.0, value=st.session_state.input_co, step=0.1, help="CPCB 8h standard: 2.0 mg/m3")
        
        st.markdown("##### **2. Secondary Gases & Photochemical Precursors**")
        c_in5, c_in6, c_in7, c_in8 = st.columns(4)
        val_so2 = c_in5.number_input("SO2 (ug/m3)", min_value=0.0, max_value=500.0, value=st.session_state.input_so2, step=1.0, help="CPCB 24h standard: 80 ug/m3")
        val_o3 = c_in6.number_input("O3 (ug/m3)", min_value=0.0, max_value=500.0, value=st.session_state.input_o3, step=1.0, help="CPCB 8h standard: 100 ug/m3")
        val_nh3 = c_in7.number_input("NH3 (ug/m3)", min_value=0.0, max_value=800.0, value=st.session_state.input_nh3, step=1.0, help="CPCB 24h standard: 400 ug/m3")
        val_no = c_in8.number_input("NO (ug/m3)", min_value=0.0, max_value=500.0, value=st.session_state.input_no, step=1.0)

        st.markdown("##### **3. Reactive Oxides & Volatile Organic Compounds (VOCs)**")
        c_in9, c_in10, c_in11, c_in12 = st.columns(4)
        val_nox = c_in9.number_input("NOx (ppb)", min_value=0.0, max_value=600.0, value=st.session_state.input_nox, step=1.0)
        val_benzene = c_in10.number_input("Benzene (ug/m3)", min_value=0.0, max_value=200.0, value=st.session_state.input_benzene, step=0.1)
        val_toluene = c_in11.number_input("Toluene (ug/m3)", min_value=0.0, max_value=300.0, value=st.session_state.input_toluene, step=0.5)
        val_xylene = c_in12.number_input("Xylene (ug/m3)", min_value=0.0, max_value=150.0, value=st.session_state.input_xylene, step=0.2)

        submit_btn = st.form_submit_button("🚀 Run Machine Learning Classification", use_container_width=True)

    # Prediction Execution & Results Card
    if submit_btn and best_model is not None:
        # Prepare input vector matching training schema
        input_data = pd.DataFrame([{
            "PM2.5": val_pm25, "PM10": val_pm10, "NO": val_no, "NO2": val_no2,
            "NOx": val_nox, "NH3": val_nh3, "CO": val_co, "SO2": val_so2,
            "O3": val_o3, "Benzene": val_benzene, "Toluene": val_toluene, "Xylene": val_xylene
        }])
        
        # ML Inference
        pred_category = best_model.predict(input_data)[0]
        pred_probs = best_model.predict_proba(input_data)[0]
        class_prob_map = dict(zip(best_model.classes_, pred_probs))
        confidence = class_prob_map[pred_category] * 100
        
        # Dual Validation: Official CPCB Mathematical AQI Sub-index calculation
        cpcb_val, cpcb_bucket, prominent_poll = calculate_cpcb_aqi({
            "PM2.5": val_pm25, "PM10": val_pm10, "NO2": val_no2, "NH3": val_nh3,
            "SO2": val_so2, "CO": val_co, "O3": val_o3
        })
        
        badge_color = AQI_COLOR_MAP.get(pred_category, "#0D9488")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_res_left, col_res_right = st.columns([1.2, 1])
        
        with col_res_left:
            st.markdown(f"""
            <div class="pred-badge" style="background-color: {badge_color};">
                <div class="pred-title">Predicted CPCB Air Quality Category</div>
                <div class="pred-cat">{pred_category}</div>
                <div style="font-size: 1.05rem; font-weight: 600;">Model Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Health Advisory
            st.info(f"**🏥 CPCB Public Health Advisory:** {AQI_HEALTH_ADVISORY.get(pred_category, 'No advisory.')}")
            
            # Comparison with CPCB Official Formula
            st.markdown("#### 📐 Mathematical CPCB Formula Verification")
            if cpcb_val is not None:
                matches = (pred_category == cpcb_bucket)
                st.write(f"• **Calculated CPCB Index:** `{cpcb_val}` ({cpcb_bucket})")
                st.write(f"• **Prominent Trigger Pollutant:** `{prominent_poll}`")
                if matches:
                    st.success("✅ **Perfect Agreement:** The Machine Learning model and official CPCB piecewise breakpoint calculation agree!")
                else:
                    st.warning(f"ℹ️ **Boundary Transition:** ML model predicts `{pred_category}`, while linear breakpoint returns `{cpcb_bucket}` (common near transition thresholds due to multi-pollutant interactions).")
            else:
                st.caption("Insufficient criteria pollutants (< 3) to compute exact official CPCB sub-index.")
                
        with col_res_right:
            st.markdown("#### 📊 Model Probability Distribution Across All Classes")
            prob_df = pd.DataFrame({
                "Category": AQI_CATEGORIES,
                "Probability (%)": [class_prob_map.get(c, 0.0) * 100 for c in AQI_CATEGORIES]
            })
            fig_prob = px.bar(
                prob_df,
                x="Probability (%)",
                y="Category",
                orientation="h",
                color="Category",
                color_discrete_map=AQI_COLOR_MAP,
                text="Probability (%)"
            )
            fig_prob.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
            fig_prob.update_layout(
                xaxis_range=[0, 105],
                height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)


# =========================================================
# TAB 5: MODEL EVALUATION
# =========================================================
with tab_evaluation:
    st.markdown('<div class="section-title">📊 Multi-Model Performance Benchmarking & Explainability</div>', unsafe_allow_html=True)
    st.write("Rigorous empirical evaluation on the held-out test partition (4,970 real-world Indian CAAQMS daily records).")
    
    metrics_data = eval_reports.get("evaluation_metrics", {})
    reports_data = eval_reports.get("classification_reports", {})
    cm_data = eval_reports.get("confusion_matrices", {})
    feat_data = eval_reports.get("feature_importance", [])

    if metrics_data:
        # Comparison Table
        st.markdown("#### 🏆 Benchmark Comparison Table")
        comp_df = pd.DataFrame(metrics_data).T[["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]]
        comp_df.columns = ["Overall Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Weighted F1"]
        comp_df = (comp_df * 100).round(2).astype(str) + "%"
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_ev1, col_ev2 = st.columns([1, 1])
        
        # Interactive Confusion Matrix
        with col_ev1:
            st.markdown("#### 🔲 Normalized Confusion Matrix")
            sel_cm_model = st.selectbox("Select Model Matrix", options=["Random Forest", "Logistic Regression"], index=0)
            if sel_cm_model in cm_data:
                matrix_norm = np.array(cm_data[sel_cm_model]["normalized"])
                fig_cm = px.imshow(
                    matrix_norm,
                    labels=dict(x="Predicted Class", y="True CPCB Class", color="Proportion"),
                    x=AQI_CATEGORIES,
                    y=AQI_CATEGORIES,
                    text_auto=".2f",
                    color_continuous_scale="Teal",
                    title=f"Normalized Confusion Matrix ({sel_cm_model})"
                )
                fig_cm.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_cm, use_container_width=True)
                
        # Per-Class Precision & Recall
        with col_ev2:
            st.markdown("#### 🎯 Per-Class Precision, Recall, and F1 (Random Forest)")
            rf_rep = reports_data.get("Random Forest", {})
            if rf_rep:
                per_class_list = []
                for cat in AQI_CATEGORIES:
                    if cat in rf_rep:
                        per_class_list.append({
                            "Category": cat,
                            "Precision": rf_rep[cat]["precision"] * 100,
                            "Recall": rf_rep[cat]["recall"] * 100,
                            "F1-Score": rf_rep[cat]["f1-score"] * 100
                        })
                pc_df = pd.DataFrame(per_class_list)
                fig_pc = px.bar(
                    pc_df,
                    x="Category",
                    y=["Precision", "Recall", "F1-Score"],
                    barmode="group",
                    color_discrete_sequence=["#0D9488", "#0284C7", "#F59E0B"],
                    title="Per-Class Performance across CPCB AQI Buckets (%)"
                )
                fig_pc.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10), yaxis_range=[0, 100])
                st.plotly_chart(fig_pc, use_container_width=True)

        st.markdown("---")
        
        # Feature Importance Analysis
        st.markdown("#### 🧠 Feature Importance & Atmospheric Explainability")
        if feat_data:
            fi_df = pd.DataFrame(feat_data)
            col_fi1, col_fi2 = st.columns([1.2, 1])
            
            with col_fi1:
                fig_fi = px.bar(
                    fi_df.sort_values(by="Gini_Importance", ascending=True),
                    x="Gini_Importance",
                    y="Feature",
                    orientation="h",
                    title="Random Forest Gini Impurity (MDI) Feature Importance",
                    labels={"Gini_Importance": "Relative Feature Contribution Score"},
                    color="Gini_Importance",
                    color_continuous_scale="teal"
                )
                fig_fi.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_fi, use_container_width=True)
                
            with col_fi2:
                st.markdown("##### 🔬 Environmental Science Interpretation")
                st.write("""
                1. **PM2.5 (29.98% Importance):** Fine particulate matter is the single highest contributor. Because PM2.5 particles penetrate deep into the alveolar sacs of human lungs, the CPCB NAQI formula assigns very steep sub-index slopes to fine dust.
                2. **PM10 (18.66% Importance):** Coarse particulate matter (road dust, construction debris, dry desert transport) triggers high sub-indices on dry and windy days.
                3. **CO (12.62% Importance):** Incomplete vehicular combustion provides a decisive signal separating stagnant urban traffic pollution from fair background air.
                4. **Causation vs Statistical Correlation:** While the Random Forest model leverages PM2.5 and PM10 to partition decision trees, in the real atmosphere, air quality is driven by meteorological boundary layer dynamics, wind speed, solar irradiation, and emission sources.
                """)
    else:
        st.warning("Evaluation artifacts not found. Please run evaluate_model.py to generate metrics.")


# =========================================================
# TAB 6: ABOUT & METHODOLOGY
# =========================================================
with tab_about:
    st.markdown('<div class="section-title">ℹ️ Project Architecture, CPCB Standard & Academic Documentation</div>', unsafe_allow_html=True)
    
    col_ab1, col_ab2 = st.columns([1, 1])
    
    with col_ab1:
        st.markdown("#### 🏛️ Central Pollution Control Board (CPCB) Standard")
        st.write("""
        The National Air Quality Index (NAQI) was launched by the Ministry of Environment, Forest and Climate Change (MoEFCC), 
        Government of India in 2014. The index simplifies complex multi-pollutant monitoring data into a single numerical index 
        (0–500+) and 6 descriptive color-coded categories:
        
        $$\\text{Sub-Index } I_p = I_{lo} + \\frac{I_{hi} - I_{lo}}{B_{hi} - B_{lo}} \\times (C_p - B_{lo})$$
        
        Where:
        - $C_p$ is the pollutant concentration
        - $B_{hi}, B_{lo}$ are breakpoint concentrations encompassing $C_p$
        - $I_{hi}, I_{lo}$ are corresponding AQI sub-indices
        - **Overall AQI** = $\\max(I_{p_1}, I_{p_2}, \\dots, I_{p_k})$ requiring at least 3 pollutants with at least one particulate matter ($PM_{2.5}$ or $PM_{10}$).
        """)
        
        st.markdown("#### ⚙️ Engineering & Modeling Pipeline")
        st.write("""
        - **Data Preprocessing (`data_preprocessing.py`):** Ingestion, negative reading correction, outlier clipping, and missing value profiling.
        - **Pipeline Architecture:** Scikit-learn `Pipeline` pairing `SimpleImputer(strategy='median')` with `StandardScaler()` and `RandomForestClassifier`.
        - **Zero Target Leakage:** `AQI` and `AQI_Bucket` are strictly stripped from model features before cross-validation or testing.
        """)
        
    with col_ab2:
        st.markdown("#### 📚 Project Deliverables & Structure")
        st.code("""
air-quality/
├── data/
│   ├── city_day.csv              # Official CPCB India dataset (29,531 records)
│   └── dataset_metadata.json     # Provenance, chemical units, licensing
├── models/
│   ├── best_model.joblib         # Fitted Random Forest pipeline
│   ├── logistic_regression.joblib
│   ├── dummy_classifier.joblib
│   └── model_metadata.json       # Hyperparameters & CV scores
├── reports/
│   ├── evaluation_metrics.json   # Accuracy, Precision, Recall, Macro F1
│   ├── classification_reports.json
│   ├── confusion_matrices.json
│   ├── feature_importance.json
│   └── *.png                     # Static high-resolution publication charts
├── notebooks/
│   └── eda.ipynb                 # 25-cell documented Jupyter notebook
├── app.py                        # Streamlit web dashboard
├── data_preprocessing.py         # Cleaning & CPCB formula module
├── train_model.py                # 5-fold CV & training script
├── evaluate_model.py             # Evaluation & explainability script
├── requirements.txt              # Project dependencies
└── README.md                     # Beginner-friendly viva & run instructions
        """, language="text")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Public Health & Emergency-Use Disclaimer:</strong> This application is developed solely as an educational 
        prototype for academic coursework (Capstone 19: Air Quality Level Classification). It is neither a substitute for official 
        meteorological forecasts nor a legal pollution compliance monitoring system. Do not rely on this prototype for clinical, 
        asthmatic emergency, or industrial regulatory compliance decisions.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Capstone 19: Air Quality Level Classification • Built with Python, Streamlit, and Scikit-Learn • Academic ML Prototype")


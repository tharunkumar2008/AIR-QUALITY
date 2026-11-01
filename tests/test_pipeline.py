"""
Unit and Integration Tests for Capstone 19 - Air Quality Level Classification
"""

import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd

from data_preprocessing import (
    AirQualityPreprocessor,
    calculate_sub_index,
    calculate_cpcb_aqi,
    POLLUTANT_FEATURES,
    AQI_CATEGORIES
)


def test_cpcb_sub_index_calculation():
    """Verify official CPCB piecewise sub-index calculations for standard breakpoints."""
    # PM2.5 tests: Breakpoints are 0-30 -> 0-50, 30-60 -> 51-100, etc.
    assert calculate_sub_index("PM2.5", 15.0) == 25.0
    assert round(calculate_sub_index("PM2.5", 45.0)) == 75
    assert round(calculate_sub_index("PM2.5", 75.0)) == 150
    assert round(calculate_sub_index("PM2.5", 105.0)) == 250
    assert round(calculate_sub_index("PM2.5", 185.0)) == 350
    
    # Negative value handling
    assert calculate_sub_index("PM2.5", -10.0) is None
    # None handling
    assert calculate_sub_index("PM2.5", None) is None


def test_overall_cpcb_aqi_logic():
    """Verify multi-pollutant CPCB AQI rules (requires at least 3 pollutants with at least one PM)."""
    # Valid readings
    readings = {
        "PM2.5": 25.0,  # sub-index: ~42 (Good)
        "PM10": 40.0,   # sub-index: ~40 (Good)
        "NO2": 30.0,    # sub-index: ~38 (Good)
        "SO2": 20.0     # sub-index: ~25 (Good)
    }
    aqi, bucket, prominent = calculate_cpcb_aqi(readings)
    assert aqi is not None
    assert bucket == "Good"
    assert prominent in ["PM2.5", "PM10"]

    # Severe case
    readings_severe = {
        "PM2.5": 300.0, # sub-index: >400 (Severe)
        "PM10": 450.0,
        "NO2": 150.0
    }
    aqi_s, bucket_s, prom_s = calculate_cpcb_aqi(readings_severe)
    assert aqi_s >= 401
    assert bucket_s == "Severe"

    # Missing PM rule violation: should return None
    readings_no_pm = {"NO2": 40.0, "SO2": 30.0, "CO": 1.0}
    aqi_fail, b_fail, p_fail = calculate_cpcb_aqi(readings_no_pm)
    assert aqi_fail is None

    # Less than 3 pollutants rule violation: should return None
    readings_under_3 = {"PM2.5": 40.0, "NO2": 20.0}
    aqi_fail2, _, _ = calculate_cpcb_aqi(readings_under_3)
    assert aqi_fail2 is None


def test_preprocessor_leakage_prevention():
    """Ensure that feature matrix X strictly excludes target and leakage columns."""
    sample_df = pd.DataFrame({
        "City": ["Delhi", "Mumbai"],
        "Date": ["2019-01-01", "2019-01-02"],
        "PM2.5": [50.0, 80.0],
        "PM10": [100.0, 150.0],
        "NO2": [25.0, 45.0],
        "CO": [1.0, 1.5],
        "AQI": [95.0, 160.0],
        "AQI_Bucket": ["Satisfactory", "Moderate"]
    })
    
    preprocessor = AirQualityPreprocessor()
    X, y = preprocessor.prepare_features_and_target(sample_df)
    
    assert "AQI" not in X.columns
    assert "AQI_Bucket" not in X.columns
    assert "City" not in X.columns
    assert "Date" not in X.columns
    assert list(y) == ["Satisfactory", "Moderate"]


def test_trained_model_loading_and_inference():
    """Verify that saved best model can load and predict on a sample feature vector."""
    model_path = os.path.join("models", "best_model.joblib")
    assert os.path.exists(model_path), "best_model.joblib must exist"
    
    model = joblib.load(model_path)
    
    # Create test sample with all 12 pollutant features
    sample_features = pd.DataFrame([{
        "PM2.5": 45.0, "PM10": 85.0, "NO": 10.0, "NO2": 25.0,
        "NOx": 20.0, "NH3": 15.0, "CO": 0.8, "SO2": 10.0,
        "O3": 30.0, "Benzene": 1.5, "Toluene": 4.0, "Xylene": 1.0
    }])
    
    prediction = model.predict(sample_features)[0]
    probabilities = model.predict_proba(sample_features)[0]
    
    assert prediction in AQI_CATEGORIES
    assert len(probabilities) == len(AQI_CATEGORIES)
    assert np.isclose(sum(probabilities), 1.0)


def test_evaluation_reports_exist():
    """Verify that all structured evaluation JSON and plots were generated."""
    req_files = [
        os.path.join("reports", "evaluation_metrics.json"),
        os.path.join("reports", "classification_reports.json"),
        os.path.join("reports", "confusion_matrices.json"),
        os.path.join("reports", "feature_importance.json"),
        os.path.join("reports", "model_comparison.png"),
        os.path.join("reports", "feature_importance.png")
    ]
    for rf in req_files:
        assert os.path.exists(rf), f"Report file {rf} is missing"

    with open(os.path.join("reports", "evaluation_metrics.json")) as f:
        metrics = json.load(f)
        assert "Random Forest" in metrics
        assert metrics["Random Forest"]["accuracy"] > 0.75
        assert metrics["Random Forest"]["macro_f1"] > 0.70


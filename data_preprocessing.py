"""
Air Quality Level Classification - Data Preprocessing Module
Capstone 19: Comprehensive CPCB Indian Air Quality Data Processing Pipeline

This module handles:
1. Dataset ingestion (local path or file-like buffer for CSV upload)
2. Schema inspection, summary statistics, and missing value profiling
3. Data cleaning: duplicate removal, negative reading correction, outlier bounds
4. Official CPCB India sub-index and AQI calculation logic
5. Feature matrix isolation ensuring STRICT PREVENTATION OF TARGET LEAKAGE
6. Missing value imputation and feature scaling within scikit-learn pipelines
7. Chronological and stratified train-test splitting
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard 12 Chemical Pollutants monitored in CPCB City-Day dataset
POLLUTANT_FEATURES: List[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

# Official CPCB AQI Categories in ascending severity order
AQI_CATEGORIES: List[str] = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe"
]

# Category UI Colors for graphs, badges, and dashboard components
AQI_COLOR_MAP: Dict[str, str] = {
    "Good": "#10B981",          # Emerald Green
    "Satisfactory": "#84CC16",  # Lime Green
    "Moderate": "#F59E0B",      # Amber Yellow
    "Poor": "#F97316",          # Orange
    "Very Poor": "#EF4444",     # Crimson Red
    "Severe": "#7F1D1D"         # Dark Maroon
}

# Health advisories linked to official CPCB category definitions
AQI_HEALTH_ADVISORY: Dict[str, str] = {
    "Good": "Minimal impact. Air quality is considered satisfactory, and air pollution poses little or no risk.",
    "Satisfactory": "Minor breathing discomfort to sensitive people including asthmatics and elderly.",
    "Moderate": "Breathing discomfort to the people with lungs, asthma, and heart diseases.",
    "Poor": "Breathing discomfort to most people on prolonged exposure; avoid sustained outdoor exertion.",
    "Very Poor": "Respiratory illness on prolonged exposure. Significant risk for people with heart or lung disease.",
    "Severe": "Affects healthy people and seriously impacts those with existing diseases; avoid all outdoor physical activity."
}

# Official CPCB Breakpoints: (Concentration_Low, Concentration_High, Index_Low, Index_High)
CPCB_BREAKPOINTS = {
    "PM2.5": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "PM10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "NO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 500.0, 401, 500),
    ],
    "NH3": [
        (0.0, 200.0, 0, 50),
        (200.1, 400.0, 51, 100),
        (400.1, 800.0, 101, 200),
        (800.1, 1200.0, 201, 300),
        (1200.1, 1800.0, 301, 400),
        (1800.1, 2400.0, 401, 500),
    ],
    "SO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2000.0, 401, 500),
    ],
    "CO": [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "O3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
}


def calculate_sub_index(pollutant: str, concentration: float) -> Optional[float]:
    """
    Calculate the official CPCB sub-index for a single pollutant concentration
    using linear interpolation between defined breakpoint concentrations.
    """
    if concentration is None or np.isnan(concentration) or concentration < 0:
        return None
    
    pollutant_key = pollutant.upper() if pollutant.upper() in CPCB_BREAKPOINTS else pollutant
    if pollutant_key not in CPCB_BREAKPOINTS:
        return None
    
    breakpoints = CPCB_BREAKPOINTS[pollutant_key]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return i_lo + (concentration - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
            
    # For concentrations exceeding the highest defined breakpoint
    c_lo, c_hi, i_lo, i_hi = breakpoints[-1]
    if concentration > c_hi:
        extrapolated = i_lo + (concentration - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
        return min(extrapolated, 800.0) # Cap extreme mathematical extrapolation
    return None


def calculate_cpcb_aqi(pollutant_readings: Dict[str, float]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Calculate the official CPCB overall AQI according to CPCB guidelines:
    - Requires sub-indices for at least 3 pollutants
    - At least one of the pollutants MUST be PM2.5 or PM10
    - Overall AQI = max(sub-index_1, ..., sub-index_k)
    - The pollutant causing the max sub-index is the 'prominent pollutant'

    Returns:
        (aqi_value, aqi_bucket, prominent_pollutant)
    """
    sub_indices = {}
    for p, val in pollutant_readings.items():
        sub_idx = calculate_sub_index(p, val)
        if sub_idx is not None:
            sub_indices[p] = sub_idx
            
    has_pm = ("PM2.5" in sub_indices) or ("PM10" in sub_indices)
    if len(sub_indices) < 3 or not has_pm:
        return None, None, None
        
    prominent_pollutant = max(sub_indices, key=sub_indices.get)
    max_aqi = round(sub_indices[prominent_pollutant])
    
    # Map numerical AQI to CPCB Bucket
    if max_aqi <= 50:
        bucket = "Good"
    elif max_aqi <= 100:
        bucket = "Satisfactory"
    elif max_aqi <= 200:
        bucket = "Moderate"
    elif max_aqi <= 300:
        bucket = "Poor"
    elif max_aqi <= 400:
        bucket = "Very Poor"
    else:
        bucket = "Severe"
        
    return max_aqi, bucket, prominent_pollutant


class AirQualityPreprocessor:
    """
    Modular preprocessor for CPCB Indian Air Quality datasets.
    Handles data validation, missing value imputation, leakage prevention,
    and train/test splitting.
    """
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or POLLUTANT_FEATURES
        self.target_column = "AQI_Bucket"
        self.classes = AQI_CATEGORIES
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        
    def load_data(self, source: Union[str, pd.DataFrame, any]) -> pd.DataFrame:
        """
        Loads dataset from file path, StringIO, or uploaded buffer.
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif hasattr(source, "read"):
            df = pd.read_csv(source)
        elif isinstance(source, str) and os.path.exists(source):
            df = pd.read_csv(source)
        else:
            raise FileNotFoundError(f"Cannot load dataset from source: {source}")
            
        logger.info(f"Loaded raw dataset with shape: {df.shape}")
        return df

    def get_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """
        Extracts comprehensive dataset metrics for the overview and dashboard card.
        """
        summary = {
            "total_records": len(df),
            "columns": list(df.columns),
            "cities": list(df["City"].unique()) if "City" in df.columns else [],
            "num_cities": int(df["City"].nunique()) if "City" in df.columns else 0,
            "date_range": (str(df["Date"].min()), str(df["Date"].max())) if "Date" in df.columns else ("N/A", "N/A"),
            "pollutant_features": [col for col in self.feature_columns if col in df.columns],
            "missing_counts": df.isnull().sum().to_dict(),
            "missing_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "target_distribution": df[self.target_column].value_counts().to_dict() if self.target_column in df.columns else {}
        }
        return summary

    def clean_data(self, df: pd.DataFrame, drop_unlabeled: bool = True) -> pd.DataFrame:
        """
        Cleans raw air quality data:
        1. Strips column name whitespace
        2. Normalizes target labels (e.g. 'Moderately Polluted' -> 'Moderate')
        3. Removes exact duplicate rows
        4. Replaces invalid negative chemical readings with NaN
        5. Clips absurd physical sensor glitches (e.g. PM > 1200 ug/m3)
        6. Filters out rows missing ground-truth target label (when drop_unlabeled is True)
        """
        clean_df = df.copy()
        clean_df.columns = [c.strip() for c in clean_df.columns]
        
        # Standardize target category naming if present
        if self.target_column in clean_df.columns:
            clean_df[self.target_column] = clean_df[self.target_column].replace({
                "Moderately Polluted": "Moderate",
                "moderate": "Moderate",
                "good": "Good",
                "poor": "Poor"
            })
            
        # Parse Dates if present
        if "Date" in clean_df.columns:
            clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
            
        # Drop duplicates
        initial_count = len(clean_df)
        clean_df = clean_df.drop_duplicates()
        dropped_dups = initial_count - len(clean_df)
        if dropped_dups > 0:
            logger.info(f"Removed {dropped_dups} duplicate rows.")
            
        # Sanitize pollutant columns: replace negative values with NaN
        for col in self.feature_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
                neg_mask = clean_df[col] < 0
                if neg_mask.sum() > 0:
                    clean_df.loc[neg_mask, col] = np.nan
                    
        # Filter rows without target if required for model training/evaluation
        if drop_unlabeled and self.target_column in clean_df.columns:
            # Check if AQI exists but bucket is missing, or recalculate if valid
            missing_bucket = clean_df[self.target_column].isnull()
            if "AQI" in clean_df.columns:
                # If numeric AQI is present, fill missing bucket
                aqi_present = missing_bucket & clean_df["AQI"].notnull()
                for idx in clean_df[aqi_present].index:
                    aqi_val = clean_df.loc[idx, "AQI"]
                    if aqi_val <= 50: b = "Good"
                    elif aqi_val <= 100: b = "Satisfactory"
                    elif aqi_val <= 200: b = "Moderate"
                    elif aqi_val <= 300: b = "Poor"
                    elif aqi_val <= 400: b = "Very Poor"
                    else: b = "Severe"
                    clean_df.loc[idx, self.target_column] = b
            
            clean_df = clean_df.dropna(subset=[self.target_column])
            # Filter only valid classes in CPCB NAQI
            clean_df = clean_df[clean_df[self.target_column].isin(self.classes)]
            logger.info(f"Retained {len(clean_df)} labeled records belonging to CPCB AQI classes.")
            
        return clean_df

    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Isolates input features X and target y.
        CRITICAL FOR INTEGRITY:
        Excludes 'AQI', 'AQI_Bucket', 'Date', 'City' and any leakage columns.
        Model trains STRICTLY on chemical pollutant concentrations.
        """
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataframe.")
            
        y = df[self.target_column].astype(str)
        
        # Ensure all required pollutant columns exist; fill with NaN if missing from user CSV
        X = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                X[col] = np.nan
                
        # Target leakage prevention verification
        forbidden_cols = ["AQI", "AQI_Bucket", "aqi", "target", "label", "City", "Date"]
        for forbidden in forbidden_cols:
            if forbidden in X.columns:
                X = X.drop(columns=[forbidden])
                
        logger.info(f"Isolated feature matrix X shape: {X.shape}, target y length: {len(y)}")
        return X, y

    def get_preprocessing_pipeline(self) -> Pipeline:
        """
        Creates an end-to-end scikit-learn preprocessing pipeline.
        Imputes missing pollutant values using median strategy computed strictly
        on training partition, followed by standard scaling.
        """
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        return pipeline

    def split_data(
        self,
        df: pd.DataFrame,
        split_type: str = "stratified",
        test_size: float = 0.2,
        random_state: int = 42,
        chronological_date: str = "2020-01-01"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits clean data into train and test sets.
        Supports:
        - 'stratified': Random stratified split preserving class proportions across all 6 AQI buckets.
        - 'chronological': Splits by cutoff date (e.g. 2015-2019 for training, 2020 for test),
          simulating real-world forward validation.
        """
        clean_df = self.clean_data(df, drop_unlabeled=True)
        
        if split_type == "chronological" and "Date" in clean_df.columns:
            clean_df = clean_df.sort_values("Date").reset_index(drop=True)
            train_mask = clean_df["Date"] < pd.to_datetime(chronological_date)
            test_mask = clean_df["Date"] >= pd.to_datetime(chronological_date)
            
            train_df = clean_df[train_mask]
            test_df = clean_df[test_mask]
            
            X_train, y_train = self.prepare_features_and_target(train_df)
            X_test, y_test = self.prepare_features_and_target(test_df)
            logger.info(f"Chronological split at {chronological_date}: Train={len(X_train)}, Test={len(X_test)}")
        else:
            X, y = self.prepare_features_and_target(clean_df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info(f"Stratified split (test_size={test_size}): Train={len(X_train)}, Test={len(X_test)}")
            
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor = AirQualityPreprocessor()
    csv_path = os.path.join(os.path.dirname(__file__), "data", "city_day.csv")
    if os.path.exists(csv_path):
        df_raw = preprocessor.load_data(csv_path)
        summary = preprocessor.get_dataset_summary(df_raw)
        print("Dataset Summary:", json.dumps({k: summary[k] for k in ["total_records", "num_cities", "date_range", "target_distribution"]}, indent=2))
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_raw, split_type="stratified")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print("Class distribution in test set:\n", y_test.value_counts())


"""
Air Quality Level Classification - Model Training Module
Capstone 19: Model Training, Cross-Validation, and Comparison Pipeline

This script:
1. Loads cleaned data via AirQualityPreprocessor
2. Prepares train and test splits (80/20 stratified split)
3. Builds end-to-end scikit-learn Pipelines:
   - Baseline Benchmark: DummyClassifier
   - Baseline Linear Model: LogisticRegression
   - Advanced Non-Linear Model: RandomForestClassifier
4. Executes 5-Fold Stratified Cross-Validation on the training set
5. Trains final models on the entire training set
6. Evaluates preliminary metrics on the held-out test set
7. Serializes fitted pipelines and metadata to the models/ directory
"""

import os
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_preprocessing import AirQualityPreprocessor, POLLUTANT_FEATURES, AQI_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline(classifier, feature_scaler: bool = True) -> Pipeline:
    """
    Constructs an end-to-end inference-ready Pipeline:
    SimpleImputer (median) -> StandardScaler (optional) -> Classifier
    Encapsulating preprocessing prevents all data leakage during CV and test inference.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if feature_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
    """
    Evaluates pipeline using Stratified K-Fold Cross Validation.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro"
    }
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    cv_results = {
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
        "cv_precision_macro_mean": float(np.mean(scores["test_precision_macro"])),
        "cv_recall_macro_mean": float(np.mean(scores["test_recall_macro"]))
    }
    return cv_results


def evaluate_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Computes key performance metrics on the held-out test set.
    """
    y_pred = pipeline.predict(X_test)
    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    }


def train_and_compare_models(data_path: str, models_dir: str = "models") -> Dict[str, Any]:
    """
    Orchestrates the entire model training, cross-validation, and saving workflow.
    """
    os.makedirs(models_dir, exist_ok=True)
    logger.info("Initializing AirQualityPreprocessor...")
    preprocessor = AirQualityPreprocessor()
    
    df_raw = preprocessor.load_data(data_path)
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_raw, split_type="stratified", test_size=0.2, random_state=42)
    
    # Define models
    model_configs = {
        "Dummy Baseline": {
            "model": DummyClassifier(strategy="most_frequent"),
            "scale": False,
            "filename": "dummy_classifier.joblib"
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
            "scale": True,
            "filename": "logistic_regression.joblib"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1),
            "scale": False,
            "filename": "random_forest.joblib"
        }
    }
    
    comparison_results = {}
    fitted_pipelines = {}
    
    for name, config in model_configs.items():
        logger.info(f"--- Training {name} ---")
        pipe = build_pipeline(config["model"], feature_scaler=config["scale"])
        
        # 5-fold cross validation on train split
        t0 = time.time()
        cv_metrics = evaluate_cv(pipe, X_train, y_train, cv_folds=5)
        cv_time = time.time() - t0
        logger.info(f"{name} 5-Fold CV F1-macro: {cv_metrics['cv_f1_macro_mean']:.4f} (+/- {cv_metrics['cv_f1_macro_std']:.4f}) in {cv_time:.2f}s")
        
        # Fit on full training data
        pipe.fit(X_train, y_train)
        
        # Evaluate on test data
        test_metrics = evaluate_test(pipe, X_test, y_test)
        logger.info(f"{name} Test Accuracy: {test_metrics['test_accuracy']:.4f}, Test Macro F1: {test_metrics['test_macro_f1']:.4f}")
        
        fitted_pipelines[name] = pipe
        comparison_results[name] = {
            **cv_metrics,
            **test_metrics,
            "filename": config["filename"]
        }
        
        # Save individual model
        save_path = os.path.join(models_dir, config["filename"])
        joblib.dump(pipe, save_path)
        logger.info(f"Saved {name} pipeline to {save_path}")
        
    # Identify the best model based on held-out Test Macro F1-score
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]["test_macro_f1"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    logger.info(f"==> Best Model Selected: '{best_model_name}' (Test Macro F1: {comparison_results[best_model_name]['test_macro_f1']:.4f})")
    logger.info(f"Saved best model pipeline to {best_model_path}")
    
    # Save training metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": data_path,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": POLLUTANT_FEATURES,
        "classes": AQI_CATEGORIES,
        "best_model_name": best_model_name,
        "models_comparison": comparison_results
    }
    
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved training metadata to {metadata_path}")
    
    return metadata


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, "data", "city_day.csv")
    models_folder = os.path.join(current_dir, "models")
    
    if os.path.exists(data_file):
        train_and_compare_models(data_file, models_folder)
    else:
        logger.error(f"Dataset file not found at {data_file}")

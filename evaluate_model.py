"""
Air Quality Level Classification - Model Evaluation & Explainability Module
Capstone 19: Comprehensive Evaluation, Classification Reports, and Visualizations

This module:
1. Evaluates fitted models on the held-out test set
2. Computes per-class Precision, Recall, F1-score, and Support
3. Generates Confusion Matrices (Normalized & Counts)
4. Calculates Random Forest Gini Feature Importance
5. Calculates Permutation Importance on held-out test data
6. Saves static visual plots to reports/
7. Serializes structured JSON evaluation reports to reports/
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.inspection import permutation_importance

from data_preprocessing import AirQualityPreprocessor, POLLUTANT_FEATURES, AQI_CATEGORIES, AQI_COLOR_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_evaluation_reports(data_path: str, models_dir: str = "models", reports_dir: str = "reports"):
    """
    Executes full evaluation pipeline on the test set and generates artifacts.
    """
    os.makedirs(reports_dir, exist_ok=True)
    preprocessor = AirQualityPreprocessor()
    df_raw = preprocessor.load_data(data_path)
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_raw, split_type="stratified", test_size=0.2, random_state=42)
    
    # Load models
    models = {
        "Dummy Baseline": joblib.load(os.path.join(models_dir, "dummy_classifier.joblib")),
        "Logistic Regression": joblib.load(os.path.join(models_dir, "logistic_regression.joblib")),
        "Random Forest": joblib.load(os.path.join(models_dir, "random_forest.joblib"))
    }
    
    overall_metrics = {}
    class_reports = {}
    confusion_matrices = {}
    
    for name, pipe in models.items():
        logger.info(f"Evaluating {name} on test set ({len(X_test)} samples)...")
        y_pred = pipe.predict(X_test)
        
        # Overall metrics
        acc = accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        p_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        r_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        overall_metrics[name] = {
            "accuracy": float(round(acc, 4)),
            "macro_precision": float(round(p_macro, 4)),
            "macro_recall": float(round(r_macro, 4)),
            "macro_f1": float(round(f1_macro, 4)),
            "weighted_precision": float(round(p_weighted, 4)),
            "weighted_recall": float(round(r_weighted, 4)),
            "weighted_f1": float(round(f1_weighted, 4))
        }
        
        # Per-class classification report
        rep_dict = classification_report(
            y_test, y_pred, labels=AQI_CATEGORIES, output_dict=True, zero_division=0
        )
        class_reports[name] = rep_dict
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=AQI_CATEGORIES)
        cm_norm = confusion_matrix(y_test, y_pred, labels=AQI_CATEGORIES, normalize="true")
        confusion_matrices[name] = {
            "raw": cm.tolist(),
            "normalized": np.round(cm_norm, 4).tolist()
        }

    # Save overall metrics JSON
    metrics_path = os.path.join(reports_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(overall_metrics, f, indent=2)
    logger.info(f"Saved metrics summary to {metrics_path}")

    # Save classification reports JSON
    reports_path = os.path.join(reports_dir, "classification_reports.json")
    with open(reports_path, "w") as f:
        json.dump(class_reports, f, indent=2)
    logger.info(f"Saved classification reports to {reports_path}")

    # Save confusion matrices JSON
    cm_path = os.path.join(reports_dir, "confusion_matrices.json")
    with open(cm_path, "w") as f:
        json.dump(confusion_matrices, f, indent=2)
    logger.info(f"Saved confusion matrices to {cm_path}")

    # ------------------ Plot 1: Model Comparison Chart ------------------
    logger.info("Generating Model Comparison Chart...")
    plt.figure(figsize=(10, 6), dpi=200)
    sns.set_theme(style="whitegrid")
    
    comp_df = pd.DataFrame(overall_metrics).T[["accuracy", "macro_precision", "macro_recall", "macro_f1"]]
    comp_df.columns = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    
    ax = comp_df.plot(kind="bar", figsize=(10, 6), colormap="crest", width=0.75, edgecolor="none")
    plt.title("Model Performance Comparison (Test Set: 4,970 Samples)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.xticks(rotation=0, fontsize=11, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend(frameon=True, facecolor="white", loc="lower right", fontsize=10)
    
    # Annotate values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
        
    plt.tight_layout()
    comp_img_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(comp_img_path)
    plt.close()
    logger.info(f"Saved comparison plot to {comp_img_path}")

    # ------------------ Plot 2: Confusion Matrices ------------------
    for m_name, file_suffix in [("Random Forest", "rf"), ("Logistic Regression", "lr")]:
        logger.info(f"Generating Confusion Matrix for {m_name}...")
        plt.figure(figsize=(8, 6.5), dpi=200)
        cm_data = np.array(confusion_matrices[m_name]["normalized"])
        
        ax = sns.heatmap(
            cm_data,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=AQI_CATEGORIES,
            yticklabels=AQI_CATEGORIES,
            cbar_kws={"label": "Normalized Proportion (Recall per Class)"},
            linewidths=0.5,
            square=True
        )
        plt.title(f"Confusion Matrix (Normalized) - {m_name}", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Predicted AQI Category", fontsize=11, fontweight="bold", labelpad=8)
        plt.ylabel("True CPCB AQI Category", fontsize=11, fontweight="bold", labelpad=8)
        plt.tight_layout()
        cm_img_path = os.path.join(reports_dir, f"confusion_matrix_{file_suffix}.png")
        plt.savefig(cm_img_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {cm_img_path}")

    # ------------------ Feature Importance & Permutation Importance ------------------
    logger.info("Computing Feature Importance & Permutation Importance for Random Forest...")
    rf_pipe = models["Random Forest"]
    rf_classifier = rf_pipe.named_steps["classifier"]
    
    # 1. Gini Impurity (MDI) Importance
    gini_importance = rf_classifier.feature_importances_
    
    # 2. Permutation Importance on held-out test data
    perm_res = permutation_importance(
        rf_pipe, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=42, n_jobs=-1
    )
    perm_mean = perm_res.importances_mean
    perm_std = perm_res.importances_std

    feat_df = pd.DataFrame({
        "Feature": POLLUTANT_FEATURES,
        "Gini_Importance": gini_importance,
        "Permutation_Importance_Mean": perm_mean,
        "Permutation_Importance_Std": perm_std
    }).sort_values(by="Gini_Importance", ascending=False).reset_index(drop=True)

    feat_path = os.path.join(reports_dir, "feature_importance.json")
    feat_df.to_json(feat_path, orient="records", indent=2)
    logger.info(f"Saved feature importance data to {feat_path}")

    # Plot: Dual Feature Importance
    plt.figure(figsize=(11, 6), dpi=200)
    plt.subplot(1, 2, 1)
    sorted_gini = feat_df.sort_values(by="Gini_Importance", ascending=True)
    plt.barh(sorted_gini["Feature"], sorted_gini["Gini_Importance"], color="#0D9488", edgecolor="none")
    plt.title("Random Forest Feature Importance (MDI / Gini)", fontsize=11, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=10)

    plt.subplot(1, 2, 2)
    sorted_perm = feat_df.sort_values(by="Permutation_Importance_Mean", ascending=True)
    plt.barh(sorted_perm["Feature"], sorted_perm["Permutation_Importance_Mean"], xerr=sorted_perm["Permutation_Importance_Std"], color="#0284C7", edgecolor="none", alpha=0.9)
    plt.title("Permutation Importance on Test Set (Drop in Macro F1)", fontsize=11, fontweight="bold")
    plt.xlabel("Mean Drop in Macro F1 Score", fontsize=10)

    plt.suptitle("Air Quality Classification: Pollutant Influence Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fi_img_path = os.path.join(reports_dir, "feature_importance.png")
    plt.savefig(fi_img_path)
    plt.close()
    logger.info(f"Saved feature importance plot to {fi_img_path}")

    print("\n" + "=" * 60)
    print("           MODEL EVALUATION SUMMARY (HELD-OUT TEST SET)")
    print("=" * 60)
    summary_table = pd.DataFrame(overall_metrics).T[["accuracy", "macro_precision", "macro_recall", "macro_f1"]]
    summary_table.columns = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    print(summary_table.to_string())
    print("\nTop 5 Most Important Pollutants (Random Forest):")
    for idx, row in feat_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']}: Gini={row['Gini_Importance']:.4f}, Permutation drop={row['Permutation_Importance_Mean']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, "data", "city_day.csv")
    models_dir = os.path.join(current_dir, "models")
    reports_dir = os.path.join(current_dir, "reports")
    
    if os.path.exists(data_file) and os.path.exists(os.path.join(models_dir, "best_model.joblib")):
        generate_evaluation_reports(data_file, models_dir, reports_dir)
    else:
        logger.error("Missing dataset or trained models. Run train_model.py first.")


# Capstone 19: Air Quality Level Classification
**An End-to-End Machine Learning Intelligence System with Modern Environmental Science Web Dashboard**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License-Educational](https://img.shields.io/badge/License-Educational%20Use-green.svg)](#)

---

## 📖 Executive Summary
Air pollution represents one of the most critical public health and environmental challenges in urban centers globally. Rapid industrialization, vehicular congestion, thermal power plants, and agricultural crop residue burning periodically subject millions of citizens to hazardous atmospheric conditions.

This project delivers a **college-level, production-grade Machine Learning classification system** that evaluates ambient atmospheric chemical measurements and classifies air quality into official **Central Pollution Control Board (CPCB) India National Air Quality Index (NAQI)** categories:
* 🟢 **Good (0 – 50)**
* 🟡 **Satisfactory (51 – 100)**
* 🟠 **Moderate (101 – 200)**
* 🔴 **Poor (201 – 300)**
* 🟣 **Very Poor (301 – 400)**
* 🟤 **Severe (401 – 500+)**

Unlike naive projects that suffer from **target leakage** (accidentally predicting AQI categories from numeric AQI columns), this architecture **strictly isolates input features** to 12 chemical pollutant concentrations (`PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`), forcing the model to learn the underlying multi-pollutant environmental physics.

---

## 🏛️ Dataset Provenance & Environmental Variables
The project uses the authentic, publicly available **CPCB Indian Air Quality Dataset (City-Day)**:
* **Origin Agency:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India.
* **Archival Portal:** National Clean Air Programme (NCAP) / Open Government Data Platform India (OGD-India).
* **Collection Scope:** **29,531 daily continuous ambient monitoring records** spanning **26 major Indian urban centers** (Delhi, Bengaluru, Mumbai, Ahmedabad, Chennai, Kolkata, Hyderabad, Lucknow, Patna, Jaipur, etc.).
* **Timeline:** January 1, 2015 to July 1, 2020.
* **License & Terms:** National Data Sharing and Accessibility Policy (NDSAP) / Open Government Data License India.

### Monitored Pollutant Units:
| Pollutant | Chemical / Physical Designation | Standard Unit | Primary Ambient Source |
| :--- | :--- | :---: | :--- |
| **PM2.5** | Fine Particulate Matter ($\le 2.5\,\mu m$) | $\mu g/m^3$ | Vehicle combustion, crop residue burning, secondary aerosols |
| **PM10** | Respirable Particulate Matter ($\le 10\,\mu m$) | $\mu g/m^3$ | Construction dust, resuspension, industrial fly ash |
| **NO** | Nitric Oxide | $\mu g/m^3$ | High-temperature internal combustion engines |
| **NO2** | Nitrogen Dioxide | $\mu g/m^3$ | Traffic exhaust, diesel generators, thermal power |
| **NOx** | Total Reactive Nitrogen Oxides | $\text{ppb}$ | Atmospheric photochemical oxidation of nitrogen |
| **NH3** | Ammonia | $\mu g/m^3$ | Fertilizer application, livestock decomposition |
| **CO** | Carbon Monoxide | $\text{mg/m}^3$ | Incomplete hydrocarbon combustion, idling engines |
| **SO2** | Sulphur Dioxide | $\mu g/m^3$ | Coal-fired power stations, petroleum refining |
| **O3** | Ground-Level Ozone | $\mu g/m^3$ | Sunlight-driven photochemical smog ($\text{VOC} + \text{NOx}$) |
| **Benzene** | Benzene ($C_6H_6$) | $\mu g/m^3$ | Evaporative fuel loss, petrochemical processing |
| **Toluene** | Toluene ($C_7H_8$) | $\mu g/m^3$ | Industrial solvents, paint thinners, fuel additives |
| **Xylene** | Xylene ($C_8H_{10}$) | $\mu g/m^3$ | Chemical manufacturing, printing, rubber production |

---

## 🔬 Mathematical CPCB NAQI Formulation
The CPCB official AQI calculates a sub-index $I_p$ for each criteria pollutant using piecewise linear interpolation across predefined regulatory concentration breakpoints:

$$I_p = I_{lo} + \frac{I_{hi} - I_{lo}}{B_{hi} - B_{lo}} \times (C_p - B_{lo})$$

Where:
* $C_p$ is the measured 24-hour concentration of pollutant $p$.
* $B_{hi}, B_{lo}$ are the regulatory breakpoint limits encompassing $C_p$.
* $I_{hi}, I_{lo}$ are the corresponding AQI category sub-index bounds.
* **Overall AQI Rule:** $\text{AQI} = \max(I_1, I_2, \dots, I_k)$ requiring at least 3 pollutants, of which at least one **must** be $PM_{2.5}$ or $PM_{10}$.

---

## 📊 Empirical Model Benchmark Results
All models were trained on 19,880 records and rigorously evaluated on a held-out test partition of **4,970 samples** using stratified 5-fold cross-validation.

| Model Architecture | 5-Fold CV Macro F1 | Test Accuracy | Macro Precision | Macro Recall | Test Macro F1 | Role / Benchmark Designation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy Baseline** | $0.0874$ | $35.53\%$ | $0.0592$ | $0.1667$ | $0.0874$ | Naive Majority Class Benchmark |
| **Logistic Regression** | $0.6867 \pm 0.009$ | $75.31\%$ | $0.7530$ | $0.6706$ | $0.7000$ | Baseline Linear Classifier |
| **Random Forest (Best)** | **$0.7704 \pm 0.005$** | **$81.45\%$** | **$0.8136$** | **$0.7687$** | **$0.7880$** | **Top Non-Linear Ensemble Model** |

### Per-Class Performance (Winning Random Forest Model):
| CPCB AQI Category | Test Precision | Test Recall | Test F1-Score | Support (Held-Out Samples) |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 **Good (0–50)** | $81.6\%$ | $61.2\%$ | $0.699$ | $268$ |
| 🟡 **Satisfactory (51–100)** | $82.9\%$ | $85.5\%$ | $0.842$ | $1,645$ |
| 🟠 **Moderate (101–200)** | $81.4\%$ | $85.6\%$ | $0.835$ | $1,766$ |
| 🔴 **Poor (201–300)** | $75.1\%$ | $66.7\%$ | $0.707$ | $556$ |
| 🟣 **Very Poor (301–400)** | $79.5\%$ | $80.5\%$ | $0.800$ | $467$ |
| 🟤 **Severe (401–500+)** | **$87.6\%$** | **$81.7\%$** | **$0.846$** | $268$ |

---

## 🧠 Feature Importance & Explainability
Analysis using both **Random Forest Mean Decrease in Impurity (Gini)** and **Permutation Importance** on unseen test data demonstrates:
1. **$PM_{2.5}$ (29.98% Gini, 0.322 drop in F1):** The single most dominant factor in Indian air quality classification.
2. **$PM_{10}$ (18.66% Gini, 0.169 drop in F1):** Secondary particulate driver, heavily influential during summer dust events.
3. **$\text{CO}$ (12.62% Gini, 0.198 drop in F1):** Indicator of incomplete hydrocarbon combustion and severe local vehicular stagnation.
4. **$O_3$ (6.25% Gini, 0.048 drop in F1):** Secondary photochemical oxidant indicating daytime solar smog.
5. **Statistical Importance vs Physical Causation:** While tree splits leverage $PM_{2.5}$ thresholds, real atmospheric severity is driven by thermal boundary layer inversions, wind stagnation, and regional emissions.

---

## 📁 Project Directory Structure
```
air-quality/
├── data/
│   ├── city_day.csv              # Official CPCB India monitoring dataset (29,531 rows)
│   └── dataset_metadata.json     # Provenance, chemical definitions, units, license
├── models/
│   ├── best_model.joblib         # Fitted Random Forest end-to-end Pipeline
│   ├── logistic_regression.joblib# Baseline linear ML pipeline
│   ├── dummy_classifier.joblib   # Baseline dummy classifier pipeline
│   └── model_metadata.json       # CV scores, hyperparameters, schema specifications
├── reports/
│   ├── evaluation_metrics.json   # Test accuracy, macro F1, weighted metrics
│   ├── classification_reports.json # Full per-class precision, recall, and support
│   ├── confusion_matrices.json   # Normalized & unnormalized matrices
│   ├── feature_importance.json   # Gini & permutation importance metrics
│   ├── model_comparison.png      # Benchmark bar chart
│   ├── confusion_matrix_rf.png   # Heatmap for Random Forest
│   ├── confusion_matrix_lr.png   # Heatmap for Logistic Regression
│   └── feature_importance.png    # Dual Gini vs Permutation importance chart
├── notebooks/
│   └── eda.ipynb                 # Comprehensive 25-cell documented Jupyter notebook
├── tests/
│   └── test_pipeline.py          # Pytest unit & integration test suite (100% pass)
├── app.py                        # Modern, interactive Streamlit web dashboard
├── data_preprocessing.py         # Cleaning, CPCB formulas, leakage prevention
├── train_model.py                # 5-fold CV & model training pipeline
├── evaluate_model.py             # Metrics calculation & report generation
├── requirements.txt              # Pinned project dependencies
└── README.md                     # Comprehensive academic documentation
```

---

## 🚀 Step-by-Step Setup & Execution Guide

### 1. Prerequisites
Ensure Python 3.10, 3.11, 3.12, 3.13, or 3.14 is installed on your system.

### 2. Clone or Navigate to Directory
```bash
cd "c:\Users\THARUN KUMAR\OneDrive\Desktop\air quality"
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Run Automated Test Suite
Verify that all preprocessing functions, CPCB formulas, model loaders, and reports pass unit tests:
```bash
python -m pytest tests/ -v
```
*(All 5 test suites pass with 100% success).*

### 5. Retrain Models (Optional)
If you wish to re-run cross-validation and generate fresh model weights:
```bash
python train_model.py
```

### 6. Generate Evaluation Reports & Plots (Optional)
```bash
python evaluate_model.py
```

### 7. Launch the Interactive Web Dashboard
Run the following command in your terminal:
```bash
python -m streamlit run app.py
```
Streamlit will launch locally and automatically open in your default browser at:
```
http://localhost:8501
```

---

## 🖥️ Dashboard Features & Tour

### Tab 1: 📋 Overview
* **KPI Metrics:** Total records (29,531), Features (12), Classes (6), Test Accuracy (81.45%), Macro F1 (0.7880).
* **CPCB Class Palette:** Color-coded badges for all 6 severity bands.
* **Proportional Donut Chart:** Interactive class breakdown across 2015–2020.

### Tab 2: 📂 Dataset Explorer
* **CSV Upload Tool:** Upload any custom air quality CSV; the system automatically detects pollutant columns and displays a schema card.
* **Filter Controls:** Dynamic filtering by City and AQI Category with preview limit slider.
* **Missing Value Profiler:** Real-time missingness percentages and completeness charts.
* **Export Clean Data:** One-click button to download the sanitized, outlier-checked dataset.

### Tab 3: 📈 Air Quality Analysis
* **Time-Series Trends:** Interactive 7-day rolling pollutant graphs for major Indian cities.
* **Inter-City Ranking:** Median AQI comparison across Delhi, Bengaluru, Mumbai, Ahmedabad, Chennai, etc.
* **Correlation Heatmap:** Interactive Pearson correlation matrix between particulate and gaseous pollutants.
* **Subcontinental Seasonality:** Visualizing the winter stagnation peak vs monsoon wet-deposition cleanse.

### Tab 4: 🔮 Live AQI Prediction
* **Preset Scenarios:** Instant buttons for *"Clean Coastal / Monsoon"*, *"Typical Urban Moderate"*, *"High Traffic / Poor"*, and *"Severe Winter Smog"*.
* **Interactive Sliders & Numeric Inputs:** Sliders for all 12 chemical features with physical validation limits.
* **Color-Coded Prediction Badge:** Instant visual feedback with official CPCB category colors.
* **Dual Comparison:** Displays both the **ML Predicted Category** and the **CPCB Official Mathematical Sub-Index** side-by-side with prominent trigger pollutant identification.
* **Health Advisory:** Official medical guidance for sensitive groups and general public.

### Tab 5: 📊 Model Evaluation
* **Benchmark Table:** Comparative analysis of Dummy Baseline vs Logistic Regression vs Random Forest.
* **Interactive Confusion Matrix:** Toggle between Random Forest and Logistic Regression, view normalized proportions or raw counts.
* **Per-Class Bar Chart:** Precision, Recall, and F1 across all 6 categories.
* **Explainability Insights:** Gini Impurity and Permutation Importance bar charts.

### Tab 6: ℹ️ About & Provenance
* Mathematical formulation of CPCB NAQI.
* Source documentation, project directory tree, and academic disclaimer.

---

## 🎓 College Viva & Presentation Guide

When presenting this project to your college professors or external examiners, highlight the following key technical differentiators:

1. **Why is overall Accuracy alone insufficient for this problem?**
   * *Answer:* The dataset exhibits **class imbalance**: `Moderate` and `Satisfactory` comprise ~68% of days, while `Good` and `Severe` account for ~5% each. A naive majority classifier achieves 35% accuracy with zero predictive value for dangerous pollution episodes. Therefore, **Macro F1-Score (0.7880)** and **Per-Class Recall** are the primary evaluation criteria.
2. **How did you prevent data leakage?**
   * *Answer:* In raw CPCB datasets, the `AQI` column directly determines `AQI_Bucket`. If `AQI` were included as an input feature, the model would simply memorize the numerical thresholds rather than learning atmospheric patterns. We strictly **drop** `AQI`, `AQI_Bucket`, `City`, and `Date` from feature matrix $X$. Furthermore, missing value imputation (median) and scaling are encapsulated inside scikit-learn `Pipeline` objects so training data never peeks at test data.
3. **Why did Random Forest outperform Logistic Regression?**
   * *Answer:* The CPCB index calculation is based on **non-linear, piecewise breakpoint intervals** and taking the **maximum** across multiple pollutants. Linear decision boundaries (Logistic Regression) cannot model sharp threshold cutoffs or multi-pollutant maximum interactions without extensive manual polynomial feature engineering. Decision trees naturally perform axis-parallel orthogonal splits that replicate piecewise environmental thresholds.
4. **How do you handle missing values in real-world CAAQMS sensors?**
   * *Answer:* In production, sensors undergo maintenance or calibration. Our pipeline incorporates a `SimpleImputer` fitted on median training concentrations. When a user enters partial readings, the model leverages learned training distributions without crashing.

---

## ⚖️ Academic Disclaimer
This project is an educational engineering prototype built for college coursework demonstration. It is not an officially certified environmental compliance monitor or medical triage tool. For regulatory compliance or public emergency advisories, refer directly to the [CPCB National CAAQM Portal](https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing).

#   A I R - Q U A L I T Y  
 
# Fault Detection & Performance Analysis of Chemical Process using Machine Learning
A machine learning pipeline for fault detection, fault classification, and efficiency-loss prediction in a multi-reactor chemical process using time-series sensor data.

## Overview
This project develops an end-to-end machine learning pipeline for monitoring continuous chemical processes. The system addresses three related tasks:

Fault Detection – determine whether a fault is occurring.
Fault Classification – identify the type of fault.
Efficiency-Loss Prediction – estimate the percentage of process efficiency lost during a fault event.
The project uses a synthetic multi-reactor chemical-process time-series dataset containing sensor measurements such as temperature, pressure, flow rates, vibration, motor current, power consumption, reaction rate, conversion, selectivity, and yield.

## Key Features
Time-aware preprocessing and train/test splitting
Reactor-specific time-series feature engineering
Lag, rolling-statistic, and rate-of-change features
Recursive Feature Elimination (RFE)
XGBoost, Random Forest, LightGBM, and Isolation Forest
Imbalanced-class handling using class weighting, SMOTE, and RandomOverSampler
Hyperparameter tuning with RandomizedSearchCV
TimeSeriesSplit cross-validation
SHAP-based model explainability
Memory-efficient feature engineering for large time-series data
Methodology
### 1. Data Preprocessing
The dataset is sorted by reactor and timestamp before processing. Missing sensor values are handled within each reactor, duplicate timestamps are removed, and sensor columns are downcast to reduce memory usage.

### 2. Feature Engineering
For each of the 12 sensor channels, the pipeline generates:

1- and 2-step lag features
5- and 10-step rolling means
5- and 10-step rolling standard deviations
First-order differences
Hour-of-day, day-of-week, and weekend indicators
This produces 87 engineered features before feature selection.

### 3. Feature Selection
Recursive Feature Elimination (RFE) is applied separately for each task to select the 30 most informative features.

### 4. Fault Detection
A tuned XGBoost classifier is used for binary fault detection. Class imbalance is handled using scale_pos_weight.

An Isolation Forest is also trained as an unsupervised baseline to compare supervised and unsupervised fault detection.

### 5. Fault Classification
Fault-only observations are used for multi-class classification. The following models are compared:

Random Forest
XGBoost
LightGBM
SMOTE and RandomOverSampler are used to address severe class imbalance.

### 6. Efficiency-Loss Regression
Two regression models are trained to predict continuous efficiency loss:

Random Forest Regressor
XGBoost Regressor
Performance is evaluated using MAE, RMSE, and R².

### 7. Explainability
SHAP is used to identify the features driving model predictions, including:

Global feature importance
Per-class feature contributions
Individual prediction explanations
Results
Binary Fault Detection
The tuned XGBoost detector achieved:

## Metric	Score
Weighted F1	99.60%
Fault Recall	91%
The supervised XGBoost model also outperformed the Isolation Forest baseline.

## Efficiency-Loss Prediction
The Random Forest regressor achieved:

Metric	Score
MAE	0.092
RMSE	0.612
R²	0.956
Multi-Class Fault Classification
The tuned XGBoost classifier achieved:

## Metric	Score
Macro F1	72.41%
Balanced Accuracy	79.19%
Performance on rare fault classes remained challenging due to severe class imbalance.

## Time-Series Validation
To avoid temporal leakage, each reactor's timeline is split chronologically, with the final portion reserved as the test set.

Hyperparameter tuning and model validation use TimeSeriesSplit rather than random cross-validation.

All preprocessing, feature selection, and resampling steps are performed within the appropriate training/CV boundaries.

Technologies
Python
Pandas
NumPy
Scikit-learn
XGBoost
LightGBM
imbalanced-learn
SHAP
Matplotlib
Seaborn
Jupyter Notebook
Project Structure
.
├── chemical_process_ML_final.ipynb
├── chemical_process_timeseries.csv
└── README.md

## Limitations
The dataset used in this project is synthetic and was designed to simulate a continuous multi-reactor chemical process. Therefore, the reported performance may be cleaner than what would be expected on real industrial data.

The rarest fault classes remain difficult to classify due to limited samples. In a real deployment, additional fault data, threshold calibration, distribution-shift monitoring, and periodic retraining would be required.

## Future Work
Explore LSTM/GRU-based sequence models
Investigate transformer-based time-series anomaly detection
Calibrate fault-detection thresholds using ROC analysis
Implement online learning and periodic retraining
Investigate adaptation to previously unseen reactors
Incorporate physics-based features from reaction kinetics# fault-detection
ML-based system to detect and classify faults in industrial processes, with EDA, feature engineering, model training, and visualizations for root-cause analysis.

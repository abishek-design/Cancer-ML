# Colorectal Cancer Risk Prediction: A Machine Learning Approach 🩺📊

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green.svg)](https://shap.readthedocs.io/)

## 📖 Overview
This repository contains a complete, end-to-end Machine Learning project aimed at predicting the likelihood and risk of colorectal cancer. The analysis is built upon demographic, dietary, lifestyle, and health indicators, utilizing advanced Explanatory AI (SHAP) and rigorous hyperparameter-tuned classification models.

## 🎯 Main Goal
To accurately predict colorectal cancer risk using a combination of patient lifestyle factors and dietary habits. Furthermore, the project emphasizes **Explainable AI (XAI)**, exposing the black-box nature of the best models to provide medical professionals with interpretable insights into primary risk drivers.

## 🚀 Key Features and Pipeline Phases
This project rigorously follows a 13-phase data science pipeline:
1. **Data Loading and Inspection**: Dynamic schema resolution and comprehensive metadata analysis.
2. **Data Cleaning**: Automated median/mode imputation and percentile-based outlier capping.
3. **Exploratory Data Analysis (EDA)**: Publication-ready Matplotlib (`ggplot` style) visualizations covering target distribution, demographic, lifestyle, and dietary factors.
4. **Feature Engineering**: Feature interaction synthesis (e.g., Age-BMI interaction) and logarithmic skew-normalization.
5. **Machine Learning Pipeline**: Sklearn `ColumnTransformer` wrapping standard scaling and One-Hot Encoding.
6. **Model Training**: Evaluation of 8 unique classifiers (Logistic Regression, Random Forest, XGBoost, etc.).
7. **Cross-Validation**: 5-Fold Stratified K-Fold implementation.
8. **Hyperparameter Tuning**: Dynamic Grid Search Optimization for top-performing architectures.
9. **Model Evaluation**: Full metric extractions including Precision-Recall, ROC-AUC, F1, and Confusion Matrices.
10. **Explainable AI (SHAP)**: TreeExplainers evaluating exact feature impacts globally.
11. **Risk Factor Analysis**: Extraction of the top positive and protective drivers for colorectal cancer.
12. **Professional Dashboards**: Aggregation of model insights into consolidated diagnostic images.
13. **Final Report Generation**: Automated summary exports.

## 🏆 Key Findings
After rigorous cross-validation and hyperparameter tuning, **Random Forest** achieved the best performance:
* **Accuracy:** 92.00%
* **ROC-AUC:** 0.966
* **Precision:** 100.0%

### Top Risk Drivers Identified (SHAP Analysis)
1. **Smoking Lifestyle** (Primary Driver)
2. **Body Mass Index (BMI)**
3. **Age & BMI Interaction**
4. **Family History of CRC**
5. **Age**

## 💻 Getting Started

### Prerequisites
Ensure you have Python 3.11+ installed. Clone this repository and install the dependencies.

```bash
pip install pandas numpy matplotlib scikit-learn xgboost shap
```

### Execution
Place your dataset (`crc_dataset.csv`) in the root directory. Then, simply execute the main script:
```bash
python colorectal_cancer_prediction.py
```

## 📊 Outputs Directory
Upon execution, an `outputs/` folder is generated containing professional-grade assets:
* `11_final_dashboard.png`: A comprehensive view of ROC curves, Confusion Matrices, and SHAP influences.
* `10_shap_summary.png`: Global feature importance XAI map.
* `12_final_report.txt`: An executive textual report.
* *Various demographic and exploratory charts.*

## 🔬 Limitations & Future Work
* **Limitations**: Current models are confined by the size and variance within the localized CSV dataset.
* **Future Work**: Integration of Deep Learning frameworks and external clinical cross-validation studies.

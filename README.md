# Telecom Customer Churn Prediction – Task 02

## Objective
Build a machine learning pipeline to predict customer churn and provide interpretable insights for retention strategies.

## Dataset
Telco Customer Churn (Kaggle) – 7043 customers originally, 7032 after cleaning  
21 features (demographics, services, tenure, charges, contract, payment method, etc.)  
Target: Churn (Yes/No) – ~26.58% churn rate

## Approach
- **Data cleaning**: Converted TotalCharges to numeric, dropped 11 missing rows, mapped Churn to 0/1
- **EDA**: Visualized churn distribution, tenure, contract types, monthly charges
- **Preprocessing**: OneHotEncoding for categoricals, StandardScaler for numerics, SMOTE for class imbalance
- **Models trained**: Logistic Regression, Random Forest, XGBoost
- **Best model**: XGBoost  
  - ROC AUC: 0.8350  
  - Churn Recall: 66%  
  - Accuracy: 77%
- **Interpretability**: SHAP summary plot & dependence plot (focus on 2-year contract)

## Key Findings
- 2-year contracts → strongest churn reducer (highest SHAP & feature importance)
- 1-year contracts also protective
- Fiber optic internet + Electronic check payment → highest churn risk
- Low tenure + high monthly charges → early churn vulnerability
- Lack of Tech Support / Online Security increases risk

## Files in this repo
- `task2updated.py` → full pipeline (loading → EDA → modeling → SHAP)
- `telco_churn.csv` → dataset
- PNGs: EDA visuals, feature importance (RF & XGBoost), SHAP plots
- `client_insight_card.png` → Client Insight Card (add this after creating it)

## How to run
```bash
# 1. Install dependencies (run once)
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap

# 2. Run the pipeline
python Telecom_churn.py

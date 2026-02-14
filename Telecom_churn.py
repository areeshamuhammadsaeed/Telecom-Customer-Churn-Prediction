import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Loading data...")
df = pd.read_csv('telco_churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Convert target to 0/1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = df.drop(columns=['customerID'], errors='ignore')

print("Shape after cleaning:", df.shape)
print("Churn rate:", df['Churn'].mean().round(4) * 100, "% churn")

plt.figure(figsize=(6, 6))
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'])
plt.title('Churn Distribution')
plt.ylabel('')
plt.savefig('churn_pie.png', dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette='Set2')
plt.title('Tenure by Churn')
plt.savefig('tenure_by_churn.png', dpi=150)
plt.close()

print("EDA visuals saved.")

# 3. Preprocessing & Modeling
categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.3%}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])


def evaluate_model(model, name):
    print(f"\n=== {name} ===")

    pipe = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm, "\n")

    return pipe


# Train models
lr_pipe = evaluate_model(LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression")
rf_pipe = evaluate_model(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), "Random Forest")
xgb_pipe = evaluate_model(
    XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss'), "XGBoost")

# 4. Feature Importance

def get_top_features(pipe, name):
    model = pipe.named_steps['model']
    features = pipe.named_steps['prep'].get_feature_names_out()

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(12)

    print(f"\nTop Features - {name}:")
    print(importance)

    plt.figure(figsize=(10, 6))
    importance.plot(kind='bar', color='teal')
    plt.title(f'Top Features - {name}')
    plt.tight_layout()
    plt.savefig(f'features_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()


get_top_features(rf_pipe, "Random Forest")
get_top_features(xgb_pipe, "XGBoost")

print("\nFiles are saved.")

# 6. SHAP Interpretability (on XGBoost model)
import shap

print("\nGenerating SHAP explanations...")

# Use the fitted XGBoost pipeline
xgb_model = xgb_pipe.named_steps['model']
xgb_preprocessor = xgb_pipe.named_steps['prep']

# Transform test data (without SMOTE for explanation)
X_test_transformed = xgb_preprocessor.transform(X_test)

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

# 1. Summary plot (beeswarm) - most important
plt.figure()
shap.summary_plot(shap_values, X_test_transformed,
                  feature_names=xgb_preprocessor.get_feature_names_out(),
                  show=False)
plt.title("SHAP Summary Plot - XGBoost")
plt.tight_layout()
plt.savefig('shap_summary_xgboost.png', dpi=150)
plt.close()

plt.figure()
shap.dependence_plot("cat__Contract_Two year", shap_values, X_test_transformed,
                     feature_names=xgb_preprocessor.get_feature_names_out(),
                     show=False)
plt.title("SHAP Dependence - 2-Year Contract")
plt.savefig('shap_dependence_contract_2yr.png', dpi=150)
plt.close()

print("SHAP plots saved: shap_summary_xgboost.png & shap_dependence_contract_2yr.png")
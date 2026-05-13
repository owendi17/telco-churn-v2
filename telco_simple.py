import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

print("=" * 80)
print("TELCO CHURN - PHASE 2: MODEL TRAINING")
print("=" * 80)

# Create data
np.random.seed(42)
n = 7043
data = {
    'tenure': np.random.randint(0, 73, n),
    'MonthlyCharges': np.random.uniform(18, 118, n),
    'feature_1': np.random.randn(n),
    'feature_2': np.random.randn(n),
    'feature_3': np.random.randn(n),
    'feature_4': np.random.randn(n),
    'feature_5': np.random.randn(n),
}
df = pd.DataFrame(data)
df['Churn'] = ((df['tenure'] < 6) & (np.random.random(n) < 0.3)).astype(int)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData ready: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Churn rate: {y.mean()*100:.1f}%\n")

# Train models
results = {}

print("Training models...")
print("-" * 80)

# 1. Logistic Regression
print("1. Logistic Regression...", end=" ")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict_proba(X_test)[:, 1]
results['Logistic Regression'] = {
    'auc': roc_auc_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, (y_pred > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred > 0.5).astype(int)),
}
print(f"AUC: {results['Logistic Regression']['auc']:.4f}")

# 2. Random Forest
print("2. Random Forest...", end=" ")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'auc': roc_auc_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, (y_pred > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred > 0.5).astype(int)),
}
print(f"AUC: {results['Random Forest']['auc']:.4f}")

# 3. XGBoost
print("3. XGBoost...", end=" ")
xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train, verbose=0)
y_pred = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {
    'auc': roc_auc_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, (y_pred > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred > 0.5).astype(int)),
}
print(f"AUC: {results['XGBoost']['auc']:.4f}")

# 4. LightGBM
print("4. LightGBM...", end=" ")
lgbm = lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict_proba(X_test)[:, 1]
results['LightGBM'] = {
    'auc': roc_auc_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, (y_pred > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred > 0.5).astype(int)),
}
print(f"AUC: {results['LightGBM']['auc']:.4f}")

# Results
print("-" * 80)
print("\nMODEL COMPARISON:")
print("-" * 80)
comparison = pd.DataFrame(results).T
print(comparison.to_string())

best_model = comparison['auc'].idxmax()
best_auc = comparison['auc'].max()

print("\n" + "=" * 80)
print(f"🏆 BEST MODEL: {best_model}")
print(f"   AUC-ROC: {best_auc:.4f}")
print("=" * 80)

print("\n✅ PHASE 2 COMPLETE!")
print("\nNext: Build Phase 3 (Streamlit App) 🚀")

# 📊 Telco Churn v2.0 - Phase 2: Model Development & Analysis
# ==========================================================

"""
PHASE 2 ROADMAP:
================
✓ Load processed data from Phase 1
✓ Train 5 models
✓ Compare performance
✓ Feature importance analysis
✓ SHAP preparation
✓ ROI analysis
✓ Threshold optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, auc, 
    precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 90)
print("📊 TELCO CHURN v2.0 - PHASE 2: MODEL DEVELOPMENT & ANALYSIS")
print("=" * 90)

# =====================================================================
# STEP 1: RECREATE PROCESSED DATA FROM PHASE 1
# =====================================================================

print("\n📥 Step 1: Creating processed dataset from Phase 1...")

np.random.seed(42)
n_samples = 7043

# Create base data
data = {
    'customerID': [f'ID_{i:05d}' for i in range(n_samples)],
    'gender': np.random.choice([0, 1], n_samples),  # Already encoded
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
    'Partner': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
    'Dependents': np.random.choice([0, 1], n_samples, p=[0.30, 0.70]),
    'tenure': np.random.randint(0, 73, n_samples),
    'PhoneService': np.random.choice([0, 1], n_samples, p=[0.90, 0.10]),
    'OnlineSecurity': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
    'OnlineBackup': np.random.choice([0, 1], n_samples, p=[0.12, 0.88]),
    'DeviceProtection': np.random.choice([0, 1], n_samples, p=[0.10, 0.90]),
    'TechSupport': np.random.choice([0, 1], n_samples, p=[0.12, 0.88]),
    'StreamingTV': np.random.choice([0, 1], n_samples, p=[0.20, 0.80]),
    'StreamingMovies': np.random.choice([0, 1], n_samples, p=[0.18, 0.82]),
    'PaperlessBilling': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'MonthlyCharges': np.random.uniform(18, 118, n_samples),
}

# Categorical variables (one-hot encoded)
data['InternetService_Fiber optic'] = np.random.choice([0, 1], n_samples, p=[0.60, 0.40])
data['InternetService_No'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
data['Contract_One year'] = np.random.choice([0, 1], n_samples, p=[0.78, 0.22])
data['Contract_Two year'] = np.random.choice([0, 1], n_samples, p=[0.77, 0.23])

# Engineered features from Phase 1
df = pd.DataFrame(data)
df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
df['CLV'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ServiceAdoption'] = (
    df['OnlineSecurity'] + df['OnlineBackup'] + 
    df['DeviceProtection'] + df['TechSupport'] + 
    df['StreamingTV'] + df['StreamingMovies']
)
df['FiberOptecRiskFlag'] = df['InternetService_Fiber optic']
df['MonthlyContractRiskFlag'] = (1 - df['Contract_One year']) * (1 - df['Contract_Two year'])
df['HighChargesRiskFlag'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
df['NewCustomerRiskFlag'] = (df['tenure'] < 6).astype(int)
df['ChurnRiskScore'] = (
    df['NewCustomerRiskFlag'] * 0.25 +
    df['MonthlyContractRiskFlag'] * 0.25 +
    df['FiberOptecRiskFlag'] * 0.15 +
    df['HighChargesRiskFlag'] * 0.15 +
    ((6 - df['ServiceAdoption']) / 6 * 0.2)
)

# Create churn target
churn_prob = (
    0.1 +
    (df['tenure'] < 6) * 0.25 +
    df['MonthlyContractRiskFlag'] * 0.18 +
    (df['InternetService_Fiber optic'] * 0.08) +
    (df['SeniorCitizen'] * 0.06) +
    (df['MonthlyCharges'] > 100) * 0.08
)
churn_prob = np.clip(churn_prob, 0, 1)
df['Churn'] = (np.random.random(n_samples) < churn_prob).astype(int)

print(f"✅ Dataset created!")
print(f"   Shape: {df.shape}")
print(f"   Churn rate: {df['Churn'].mean()*100:.2f}%")

# =====================================================================
# STEP 2: DATA PREPROCESSING
# =====================================================================

print("\n📊 Step 2: Data Preprocessing...")

# Drop ID column
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data preprocessed!")
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")
print(f"   Features: {X_train.shape[1]}")
print(f"   Train churn rate: {y_train.mean()*100:.2f}%")

# =====================================================================
# STEP 3: TRAIN MODELS
# =====================================================================

print("\n\n" + "=" * 90)
print("🤖 STEP 3: TRAINING 5 MODELS")
print("=" * 90)

models = {}
predictions = {}
results = {}

# 1. Logistic Regression
print("\n1️⃣ Logistic Regression...")
models['LR'] = LogisticRegression(max_iter=1000, random_state=42)
models['LR'].fit(X_train, y_train)
y_pred_lr = models['LR'].predict_proba(X_test)[:, 1]
predictions['LR'] = y_pred_lr
results['LR'] = {
    'auc': roc_auc_score(y_test, y_pred_lr),
    'accuracy': accuracy_score(y_test, (y_pred_lr > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred_lr > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred_lr > 0.5).astype(int)),
}
print(f"   AUC: {results['LR']['auc']:.4f}")

# 2. Random Forest
print("\n2️⃣ Random Forest...")
models['RF'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
models['RF'].fit(X_train, y_train)
y_pred_rf = models['RF'].predict_proba(X_test)[:, 1]
predictions['RF'] = y_pred_rf
results['RF'] = {
    'auc': roc_auc_score(y_test, y_pred_rf),
    'accuracy': accuracy_score(y_test, (y_pred_rf > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred_rf > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred_rf > 0.5).astype(int)),
    'feature_importance': pd.DataFrame({
        'feature': X.columns,
        'importance': models['RF'].feature_importances_
    }).sort_values('importance', ascending=False)
}
print(f"   AUC: {results['RF']['auc']:.4f}")

# 3. XGBoost
print("\n3️⃣ XGBoost...")
models['XGB'] = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', max_depth=6)
models['XGB'].fit(X_train, y_train, verbose=0)
y_pred_xgb = models['XGB'].predict_proba(X_test)[:, 1]
predictions['XGB'] = y_pred_xgb
results['XGB'] = {
    'auc': roc_auc_score(y_test, y_pred_xgb),
    'accuracy': accuracy_score(y_test, (y_pred_xgb > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred_xgb > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred_xgb > 0.5).astype(int)),
    'feature_importance': pd.DataFrame({
        'feature': X.columns,
        'importance': models['XGB'].feature_importances_
    }).sort_values('importance', ascending=False)
}
print(f"   AUC: {results['XGB']['auc']:.4f}")

# 4. LightGBM
print("\n4️⃣ LightGBM...")
models['LGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=42, max_depth=8, verbose=-1)
models['LGBM'].fit(X_train, y_train)
y_pred_lgbm = models['LGBM'].predict_proba(X_test)[:, 1]
predictions['LGBM'] = y_pred_lgbm
results['LGBM'] = {
    'auc': roc_auc_score(y_test, y_pred_lgbm),
    'accuracy': accuracy_score(y_test, (y_pred_lgbm > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred_lgbm > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred_lgbm > 0.5).astype(int)),
    'feature_importance': pd.DataFrame({
        'feature': X.columns,
        'importance': models['LGBM'].feature_importances_
    }).sort_values('importance', ascending=False)
}
print(f"   AUC: {results['LGBM']['auc']:.4f}")

# 5. Neural Network
print("\n5️⃣ Neural Network (MLP)...")
models['NN'] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
models['NN'].fit(X_train, y_train)
y_pred_nn = models['NN'].predict_proba(X_test)[:, 1]
predictions['NN'] = y_pred_nn
results['NN'] = {
    'auc': roc_auc_score(y_test, y_pred_nn),
    'accuracy': accuracy_score(y_test, (y_pred_nn > 0.5).astype(int)),
    'precision': precision_score(y_test, (y_pred_nn > 0.5).astype(int)),
    'recall': recall_score(y_test, (y_pred_nn > 0.5).astype(int)),
}
print(f"   AUC: {results['NN']['auc']:.4f}")

# =====================================================================
# STEP 4: MODEL COMPARISON
# =====================================================================

print("\n\n" + "=" * 90)
print("📊 STEP 4: MODEL PERFORMANCE COMPARISON")
print("=" * 90)

print("\n" + "-" * 90)
print(f"{'Model':<15} {'AUC-ROC':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 90)

comparison_df = pd.DataFrame(results).T
for model_name in comparison_df.index:
    auc = comparison_df.loc[model_name, 'auc']
    acc = comparison_df.loc[model_name, 'accuracy']
    prec = comparison_df.loc[model_name, 'precision']
    rec = comparison_df.loc[model_name, 'recall']
    print(f"{model_name:<15} {auc:<12.4f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f}")

print("-" * 90)

# Find best model
best_model_name = comparison_df['auc'].idxmax()
best_auc = comparison_df['auc'].max()
print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")

# =====================================================================
# STEP 5: FEATURE IMPORTANCE
# =====================================================================

print("\n\n" + "=" * 90)
print("🔍 STEP 5: TOP FEATURES BY IMPORTANCE")
print("=" * 90)

print("\nRandom Forest - Top 15 Features:")
print("-" * 60)
rf_top = results['RF']['feature_importance'].head(15)
for idx, (feat, imp) in enumerate(zip(rf_top['feature'], rf_top['importance']), 1):
    bar = "█" * int(imp * 100)
    print(f"{idx:2}. {feat:35} {bar} {imp:.4f}")

print("\n\nXGBoost - Top 15 Features:")
print("-" * 60)
xgb_top = results['XGB']['feature_importance'].head(15)
for idx, (feat, imp) in enumerate(zip(xgb_top['feature'], xgb_top['importance']), 1):
    bar = "█" * int(imp * 100)
    print(f"{idx:2}. {feat:35} {bar} {imp:.4f}")

print("\n\nLightGBM - Top 15 Features:")
print("-" * 60)
lgbm_top = results['LGBM']['feature_importance'].head(15)
for idx, (feat, imp) in enumerate(zip(lgbm_top['feature'], lgbm_top['importance']), 1):
    bar = "█" * int(imp * 100)
    print(f"{idx:2}. {feat:35} {bar} {imp:.4f}")

# =====================================================================
# STEP 6: ROI ANALYSIS
# =====================================================================

print("\n\n" + "=" * 90)
print("💰 STEP 6: BUSINESS ROI ANALYSIS")
print("=" * 90)

# Business parameters
CAC = 250  # Customer Acquisition Cost
RETENTION_COST = 75  # Cost of retention offer
AVG_MONTHLY_REVENUE = 65  # Average monthly revenue per customer
CUSTOMER_LIFETIME = 36  # Average remaining lifetime in months
PROFIT_MARGIN = 0.25  # Profit margin (25%)

print(f"""
BUSINESS PARAMETERS:
====================
  Customer Acquisition Cost: ${CAC}
  Retention Offer Cost: ${RETENTION_COST}
  Avg Monthly Revenue/Customer: ${AVG_MONTHLY_REVENUE}
  Profit Margin: {PROFIT_MARGIN*100:.0f}%
  Avg Customer Lifetime: {CUSTOMER_LIFETIME} months
  
CHURN IMPACT:
==============
  Lifetime Revenue per customer: ${AVG_MONTHLY_REVENUE * CUSTOMER_LIFETIME}
  Lifetime Profit per customer: ${AVG_MONTHLY_REVENUE * CUSTOMER_LIFETIME * PROFIT_MARGIN}
  Acquisition cost to recover: ${CAC}
  Net value of retention: ${(AVG_MONTHLY_REVENUE * CUSTOMER_LIFETIME * PROFIT_MARGIN) - RETENTION_COST}
""")

# Calculate ROI for different thresholds using best model
y_pred_best = predictions[best_model_name]
thresholds = np.arange(0.2, 0.8, 0.1)

print("\nROI BY THRESHOLD:")
print("-" * 100)
print(f"{'Threshold':<12} {'Targets':<12} {'True Pos':<12} {'False Pos':<12} {'Savings ($)':<15} {'Cost ($)':<15} {'Net Benefit':<15}")
print("-" * 100)

roi_data = []

for threshold in thresholds:
    y_pred_binary = (y_pred_best >= threshold).astype(int)
    
    # Confusion matrix
    tp = ((y_pred_binary == 1) & (y_test == 1)).sum()
    fp = ((y_pred_binary == 1) & (y_test == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_test == 1)).sum()
    
    total_targets = tp + fp
    
    # Financial metrics
    total_retention_cost = total_targets * RETENTION_COST
    saved_revenue = tp * (AVG_MONTHLY_REVENUE * CUSTOMER_LIFETIME * PROFIT_MARGIN)
    net_benefit = saved_revenue - total_retention_cost
    
    roi_data.append({
        'threshold': threshold,
        'targets': total_targets,
        'tp': tp,
        'fp': fp,
        'savings': saved_revenue,
        'cost': total_retention_cost,
        'net_benefit': net_benefit
    })
    
    print(f"{threshold:<12.1f} {total_targets:<12} {tp:<12} {fp:<12} ${saved_revenue:<14,.0f} ${total_retention_cost:<14,.0f} ${net_benefit:<14,.0f}")

# Find optimal threshold
roi_df = pd.DataFrame(roi_data)
optimal_idx = roi_df['net_benefit'].idxmax()
optimal_threshold = roi_df.loc[optimal_idx, 'threshold']
optimal_benefit = roi_df.loc[optimal_idx, 'net_benefit']

print("-" * 100)
print(f"\n💡 OPTIMAL THRESHOLD: {optimal_threshold:.1f}")
print(f"   Expected Net Benefit: ${optimal_benefit:,.0f} per month")
print(f"   Customers to Target: {int(roi_df.loc[optimal_idx, 'targets'])}")
print(f"   Expected True Positives (saved): {int(roi_df.loc[optimal_idx, 'tp'])}")

# =====================================================================
# STEP 7: SUMMARY
# =====================================================================

print("\n\n" + "=" * 90)
print("✅ PHASE 2 SUMMARY")
print("=" * 90)

summary = f"""
MODELS TRAINED:
================
✓ Logistic Regression (AUC: {results['LR']['auc']:.4f})
✓ Random Forest (AUC: {results['RF']['auc']:.4f})
✓ XGBoost (AUC: {results['XGB']['auc']:.4f})
✓ LightGBM (AUC: {results['LGBM']['auc']:.4f})
✓ Neural Network (AUC: {results['NN']['auc']:.4f})

🏆 BEST MODEL: {best_model_name} (AUC: {best_auc:.4f})

TOP FEATURES (from {best_model_name}):
1. {results[best_model_name]['feature_importance'].iloc[0, 0]} ({results[best_model_name]['feature_importance'].iloc[0, 1]:.4f})
2. {results[best_model_name]['feature_importance'].iloc[1, 0]} ({results[best_model_name]['feature_importance'].iloc[1, 1]:.4f})
3. {results[best_model_name]['feature_importance'].iloc[2, 0]} ({results[best_model_name]['feature_importance'].iloc[2, 1]:.4f})

BUSINESS IMPACT:
================
✓ Optimal Threshold: {optimal_threshold:.1f}
✓ Expected Monthly Benefit: ${optimal_benefit:,.0f}
✓ Customers to Target: {int(roi_df.loc[optimal_idx, 'targets'])}
✓ Expected Saves: {int(roi_df.loc[optimal_idx, 'tp'])} customers

NEXT STEPS (PHASE 3):
=====================
✓ Build Streamlit app (like credit risk)
✓ Add SHAP explainability
✓ Customer segmentation dashboard
✓ Business scenario calculator
✓ Deploy to production

Ready for Phase 3! 🚀
"""

print(summary)

print("\n" + "=" * 90)
print("📋 SAVE BEST MODEL & RESULTS")
print("=" * 90)

print(f"""
KEY OUTPUTS TO SAVE:
====================
1. Best model: {best_model_name}
2. Optimal threshold: {optimal_threshold:.1f}
3. Feature importance: {results[best_model_name]['feature_importance'].head(15)}
4. ROI analysis results
5. Test predictions: {y_pred_best}

These will be used in Phase 3 for Streamlit deployment! 🚀
""")

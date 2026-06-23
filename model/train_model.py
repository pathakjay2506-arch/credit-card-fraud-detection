import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import pickle
import os

print("=" * 50)
print("  Credit Card Fraud Detection - Model Training")
print("=" * 50)

# ── Generate synthetic fraud dataset ──────────────
print("\n[1/5] Generating synthetic transaction data...")
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    weights=[0.97, 0.03],   # 97% legit, 3% fraud (realistic imbalance)
    random_state=42
)

feature_names = [
    "transaction_amount", "time_of_day", "merchant_category",
    "distance_from_home", "foreign_transaction", "online_transaction",
    "transaction_frequency", "account_age_days",
    "prev_fraud_flag", "device_trust_score"
]

df = pd.DataFrame(X, columns=feature_names)
df['is_fraud'] = y

print(f"  Total transactions : {len(df)}")
print(f"  Legitimate (0)     : {sum(y==0)}")
print(f"  Fraudulent  (1)    : {sum(y==1)}")

# ── Split data ─────────────────────────────────────
print("\n[2/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Apply SMOTE ────────────────────────────────────
print("\n[3/5] Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE - Class 0: {sum(y_train_sm==0)}, Class 1: {sum(y_train_sm==1)}")

# ── Scale features ─────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

# ── Train models ───────────────────────────────────
print("\n[4/5] Training models...")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_sm)
lr_preds = lr.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_sm)
rf_preds = rf.predict(X_test_scaled)

def get_metrics(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred) * 100, 2),
        "f1_score":  round(f1_score(y_true, y_pred) * 100, 2),
    }

lr_metrics = get_metrics("Logistic Regression", y_test, lr_preds)
rf_metrics = get_metrics("Random Forest",        y_test, rf_preds)

print("\n  Model Comparison:")
print(f"  {'Metric':<12} {'Logistic Reg':>15} {'Random Forest':>15}")
print("  " + "-" * 44)
for key in ["accuracy","precision","recall","f1_score"]:
    print(f"  {key:<12} {str(lr_metrics[key])+'%':>15} {str(rf_metrics[key])+'%':>15}")

# ── Save best model ────────────────────────────────
print("\n[5/5] Saving models...")
os.makedirs("model", exist_ok=True)

with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/metrics.pkl", "wb") as f:
    pickle.dump({"lr": lr_metrics, "rf": rf_metrics}, f)

print("  fraud_model.pkl  ✅ saved")
print("  scaler.pkl       ✅ saved")
print("  metrics.pkl      ✅ saved")
print("\n  Training Complete!")
print("=" * 50)
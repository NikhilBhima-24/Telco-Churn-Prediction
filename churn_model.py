# ==========================================================
# TELCO CHURN FINAL MODEL
# Logistic Regression + Class Weight + Threshold Tuning
# ==========================================================

# ------------------------------
# IMPORT LIBRARIES
# ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ------------------------------
# LOAD DATASET
# ------------------------------

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ------------------------------
# BASIC CLEANING
# ------------------------------

# Drop unnecessary column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ------------------------------
# ENCODING
# ------------------------------

df = pd.get_dummies(df, drop_first=True)

# ------------------------------
# SPLIT DATA
# ------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------
# FEATURE SCALING
# ------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# MODEL TRAINING
# ------------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ------------------------------
# THRESHOLD ADJUSTMENT
# ------------------------------

y_probs = model.predict_proba(X_test_scaled)[:, 1]

threshold = 0.4   # Business-optimized threshold
y_pred = (y_probs >= threshold).astype(int)

# ------------------------------
# MODEL EVALUATION
# ------------------------------

print("===== FINAL TUNED LOGISTIC REGRESSION =====")
print("Threshold Used:", threshold)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_probs))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------
# CONFUSION MATRIX (CLEAN)
# ------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Tuned Logistic Regression", fontsize=13)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])

# Add values inside boxes
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 fontsize=12)

plt.colorbar()
plt.tight_layout()
plt.show()

# ------------------------------
# ROC CURVE (CLEAN)
# ------------------------------

fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Tuned Logistic Regression", fontsize=13)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
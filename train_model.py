"""
train_model.py
Day 5-9: Feature Engineering, Model Training & Evaluation
----------------------------------------------------------
This script:
1. Loads and cleans the IBM HR Attrition dataset
2. Encodes categorical columns into numbers (ML models only understand numbers)
3. Splits data into training and testing sets
4. Trains a Logistic Regression model (baseline) AND a Random Forest model
5. Compares them and saves the better one to disk for the Flask app to use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib
import os

# ---------------------------------------------------------
# STEP 1: Load the raw dataset
# ---------------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Drop columns that carry no useful information (confirmed in notebook 01)
df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

# ---------------------------------------------------------
# STEP 2: Separate target (what we predict) from features (what we use to predict)
# ---------------------------------------------------------
# Attrition is currently text: "Yes" / "No". Convert to 1 / 0.
# We do this manually (not with LabelEncoder) so we KNOW for certain
# Yes=1 (left the company) and No=0 (stayed) — order matters for interpretation.
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['Attrition'])   # Features: everything except the answer
y = df['Attrition']                  # Target: the answer we want to predict

# ---------------------------------------------------------
# STEP 3: Encode categorical (text) columns into numbers
# ---------------------------------------------------------
# Models like Random Forest and Logistic Regression cannot read text like
# "Sales" or "Married" — everything must become a number.
# LabelEncoder assigns each unique text value an integer (e.g. "Sales"->2, "R&D"->1)
categorical_cols = X.select_dtypes(include='object').columns.tolist()
print("Encoding these categorical columns:", categorical_cols)

label_encoders = {}  # we save each column's encoder so the Flask app can reuse it later

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # store it, we'll need it to decode/encode new user input later

# ---------------------------------------------------------
# STEP 4: Split into training and testing sets
# ---------------------------------------------------------
# 80% of data trains the model, 20% is held back to test on UNSEEN data.
# random_state=42 makes the split reproducible (same split every time we run this).
# stratify=y ensures both sets have the same proportion of Yes/No as the original
# data — important because our target classes are imbalanced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# STEP 5: Scale numeric features (helps Logistic Regression converge properly)
# ---------------------------------------------------------
# StandardScaler transforms each numeric feature to have mean=0, std=1.
# Random Forest does NOT need this (it splits on thresholds, not distances),
# but Logistic Regression's math is distance/gradient based, so scaling
# genuinely improves its performance and training stability.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# STEP 6: Train a baseline model — Logistic Regression
# ---------------------------------------------------------
# We always start with a simple, interpretable baseline model.
# If our "fancier" model can't beat this, the fancier model isn't worth using.
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

print("\n--- Logistic Regression Results ---")
print("Accuracy :", accuracy_score(y_test, log_preds))
print("Precision:", precision_score(y_test, log_preds))
print("Recall   :", recall_score(y_test, log_preds))
print("F1 Score :", f1_score(y_test, log_preds))

from sklearn.model_selection import GridSearchCV

# Define a grid of hyperparameter combinations to try
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10, None],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# GridSearchCV tries every combination in param_grid using 5-fold cross-validation:
# it splits the TRAINING data into 5 chunks, trains on 4, validates on the 5th,
# rotates which chunk is held out, and averages the score across all 5 rounds.
# This gives a much more reliable estimate than a single train/test split.
# scoring='f1' -> since our classes are imbalanced, we optimize for F1
# (balance of precision and recall) instead of misleading plain accuracy.
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters found:", grid_search.best_params_)

# Use the best combination found as our final model
rf_model = grid_search.best_estimator_
rf_preds = rf_model.predict(X_test)
print("\n--- Random Forest Results (tuned) ---")
print("Accuracy :", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall   :", recall_score(y_test, rf_preds))
print("F1 Score :", f1_score(y_test, rf_preds))

print("\nFull classification report (Random Forest):")
print(classification_report(y_test, rf_preds))

# ---------------------------------------------------------
# STEP 8: Feature Importance — WHY does the model predict what it predicts?
# ---------------------------------------------------------
# Random Forest can tell us which features it relied on most when making
# decisions across all its trees. This is the "explainability" part of our project.
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 most influential features:")
print(importances.head(10))

# Save a bar chart of feature importance — used later in report + Flask UI
plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='barh', color='#4a90d9')
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
os.makedirs("static", exist_ok=True)
plt.savefig("static/feature_importance.png")
print("\nSaved feature importance chart to static/feature_importance.png")

# ---------------------------------------------------------
# STEP 9: Save everything the Flask app will need
# ---------------------------------------------------------
# We save:
#  - the trained model itself
#  - the scaler (not strictly needed for RF predictions, but kept for consistency
#    in case we let users switch models later)
#  - the label encoders (to convert new user text input into the same numeric
#    codes the model was trained on)
#  - the exact list/order of feature columns (the model expects inputs in this
#    exact order every time)
os.makedirs("model", exist_ok=True)
joblib.dump(rf_model, "model/attrition_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(list(X.columns), "model/feature_columns.pkl")

print("\nAll model artifacts saved inside the 'model/' folder.")
print("Random Forest is our FINAL model — used by the Flask app for predictions.")

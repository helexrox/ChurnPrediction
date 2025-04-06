import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# STEP 0: Load the preprocessed dataset (replace with your path if needed)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# === Preprocessing Block ===
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop('customerID', axis=1, inplace=True)

binary_cols = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, columns=[
    'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod'
], drop_first=True)

# === Training Block ===
X = df.drop("Churn", axis=1)
y = df["Churn"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)

# === Evaluation Block ===
print("📊 Classification Report:")
print(classification_report(y, y_pred))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y, y_pred))

import joblib

# Save the model to a file
joblib.dump(model, "customer_churn_model.pkl")
print("✅ Model saved as 'customer_churn_model.pkl'")

import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[-10:]  # Top 10 important features

# Plotting
plt.figure(figsize=(10, 6))
plt.title("Top 10 Important Features for Churn Prediction")
plt.barh(range(len(indices)), importances[indices], color='skyblue')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
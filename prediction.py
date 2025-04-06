import pandas as pd
import joblib

# Load the saved model
model = joblib.load("customer_churn_model.pkl")

# Define all columns used during training
feature_columns = model.feature_names_in_

# New customer sample (values for relevant features)
new_customer_data = {
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 12,
    'PhoneService': 1,
    'PaperlessBilling': 1,
    'MonthlyCharges': 70.0,
    'TotalCharges': 820.5,
    'OnlineSecurity': 0,
    'OnlineBackup': 1,
    'DeviceProtection': 1,
    'TechSupport': 0,
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'MultipleLines_No phone service': 0,
    'MultipleLines_Yes': 0,
    'InternetService_Fiber optic': 1,
    'InternetService_No': 0,
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'PaymentMethod_Credit card (automatic)': 0,
    'PaymentMethod_Electronic check': 1,
    'PaymentMethod_Mailed check': 0,
}

# Create DataFrame with all expected features (fill missing with 0)
new_df = pd.DataFrame([new_customer_data])
for col in feature_columns:
    if col not in new_df.columns:
        new_df[col] = 0  # add missing columns with default value

# Reorder columns to match training set
new_df = new_df[feature_columns]

# Predict
prediction = model.predict(new_df)[0]
print("🔮 Prediction:", "Churn" if prediction == 1 else "No Churn")

proba = model.predict_proba(new_df)[0][1]  # probability of churn (class 1)
print(f"💡 Churn Probability: {proba:.2f}")
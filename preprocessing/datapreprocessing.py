import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 1: Clean 'TotalCharges'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Step 2: Drop 'customerID'
df.drop('customerID', axis=1, inplace=True)

# Step 3: Encode categorical features
# Encode binary Yes/No columns
binary_cols = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})

# Encode 'gender' and 'Churn'
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode multi-class categorical columns
df = pd.get_dummies(df, columns=[
    'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod'
], drop_first=True)

# Step 4 (Optional): Normalize numerical features
scaler = StandardScaler()
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Final preprocessed dataframe
df.reset_index(drop=True, inplace=True)
print("✅ Preprocessing complete. Shape:", df.shape)
df.head()
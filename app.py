from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model/customer_churn_model.pkl")
feature_columns = model.feature_names_in_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Fetch form data and convert to appropriate types
    form = request.form
    input_data = {
        'gender': float(form['gender']),
        'SeniorCitizen': float(form['SeniorCitizen']),
        'Partner': float(form['Partner']),
        'Dependents': float(form['Dependents']),
        'tenure': float(form['tenure']),
        'PhoneService': float(form['PhoneService']),
        'PaperlessBilling': float(form['PaperlessBilling']),
        'MonthlyCharges': float(form['MonthlyCharges']),
        'TotalCharges': float(form['TotalCharges']),
        'OnlineSecurity': float(form['OnlineSecurity']),
        'OnlineBackup': float(form['OnlineBackup']),
        'DeviceProtection': float(form['DeviceProtection']),
        'TechSupport': float(form['TechSupport']),
        'StreamingTV': float(form['StreamingTV']),
        'StreamingMovies': float(form['StreamingMovies']),
        f"MultipleLines_{form['MultipleLines']}": 1,
        f"InternetService_{form['InternetService']}": 1,
        f"Contract_{form['Contract']}": 1,
        f"PaymentMethod_{form['PaymentMethod']}": 1
    }

    df = pd.DataFrame([input_data])

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    # ✅ Friendly prediction text
    prediction_text = "✅ Customer is likely to churn." if prediction == 1 else "🟢 Customer is likely to stay."

    return render_template('result.html',
                           prediction_text=prediction_text,
                           probability=f"{proba * 100:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
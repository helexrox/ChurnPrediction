# 🔮 Customer Churn Prediction Web App

A Flask-based web application that predicts whether a telecom customer is likely to churn or stay, using a trained machine learning model.

---

## 🚀 Demo

> 📍 After running the app locally, visit:  
**`http://127.0.0.1:5000`**

### 🎥 Features
- ✅ Dropdowns for categorical features (e.g. Contract, Internet Service)
- ✅ Sliders/Number inputs for numerical features (tenure, charges, etc.)
- ✅ Beautiful UI using HTML/CSS
- ✅ Predicts customer churn using a trained Random Forest model
- ✅ Displays prediction with confidence percentage

---

## 🧠 Model Overview

- **Algorithm:** Random Forest Classifier
- **Dataset:** Telco Customer Churn (IBM public dataset)
- **Features Used:**
  - Binary/Numeric:
    - `gender`, `SeniorCitizen`, `Partner`, `Dependents`
    - `PhoneService`, `PaperlessBilling`, `tenure`
    - `MonthlyCharges`, `TotalCharges`
    - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - One-Hot Encoded:
    - `MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`

---

## 📁 Project Structure

```
CustomerChurn/
│
├── model/
│   └── customer_churn_model.pkl       # Trained model file
│
├── templates/
│   ├── index.html                     # Input form page
│   └── result.html                    # Result display page
│
├── static/                            # Optional: CSS/JS files
│
├── app.py                             # Main Flask app script
├── requirements.txt                   # Required Python packages
└── README.md                          # Project documentation
```

---

## 💻 How to Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-flask.git
cd customer-churn-flask
```

### 2. Create Virtual Environment (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate      # For Mac/Linux
venv\Scripts\activate         # For Windows
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

Visit:  
`http://127.0.0.1:5000`

---

## 📄 `requirements.txt`

```txt
flask
pandas
scikit-learn
joblib
```

You can create this file with the above contents if it's not already there.

---

## 🖼 UI Screens

> (Add screenshots here if available)

- 📋 Input form with dropdowns, sliders, and number inputs
- 📊 Result page showing:
  - ✅ Churn or 🟢 No Churn
  - 💯 Confidence Percentage

---

## ✨ Prediction Output

The model will return:
- `✅ Customer is likely to churn.`  
- or  
- `🟢 Customer is likely to stay.`  

Also shows:  
- `Confidence: 82.45%` — this is the model's predicted probability of churn.

---

## 📚 Dataset Info

- Dataset: [Telco Customer Churn – IBM Sample](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Preprocessing Done:
  - Converted `TotalCharges` to numeric
  - Filled missing values
  - One-hot encoded categorical columns
  - Trained a `RandomForestClassifier`

---

## 📌 Future Enhancements

- 🔁 CSV Upload to predict multiple customer churns at once
- 📈 Churn visualizations and dashboards
- 🧠 Model tuning with feature importance graphs
- ☁️ Deploy the app on Render/Heroku for live usage
- 📱 Make it fully mobile responsive

---

## 🙋‍♂️ About Me

**Ankith**  
🎓 Computer Science (AI & ML) @ JSS Academy  
💻 Aspiring Software Developer | 💡 Machine Learning Enthusiast

📬 Connect:  
- [LinkedIn](https://www.linkedin.com/in/ankithpradeep/)  
- [GitHub](https://github.com/helexrox)

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to use, modify, and build on it.

---

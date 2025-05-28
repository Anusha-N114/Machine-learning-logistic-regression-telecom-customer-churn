# 📞 Telecom Customer Churn Prediction using Logistic Regression

This project applies Logistic Regression to predict customer churn in a telecom company. The goal is to identify customers at risk of leaving and enable proactive retention strategies.

---

## 📌 Objective

- Predict whether a customer will churn based on their service usage and demographics
- Analyze key features contributing to churn
- Build and evaluate a Logistic Regression model
- Use residual analysis and metrics to assess model performance

---

## 🧾 Dataset Overview

- CustomerID
- Gender, SeniorCitizen, Partner, Dependents
- Tenure, MonthlyCharges, TotalCharges
- Services: Phone, Internet, Streaming, etc.
- Contract, Payment Method
- Target: `Churn` (Yes/No)

---

## 📊 EDA Highlights

- Handled missing values and cleaned numerical columns
- Visualized churn distribution across services, contract types, and charges
- Identified features most correlated with churn (e.g., tenure, contract type)

---

## 🔍 Feature Engineering

- Converted categorical variables using one-hot encoding
- Scaled numerical features using StandardScaler
- Removed multicollinearity where necessary

---

## 🤖 Model Building: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

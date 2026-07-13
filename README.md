# Bank Customer Churn Prediction using Machine Learning

## Predictive Modeling & Risk Scoring for Bank Customer Churn

### Submitted By

- **Name:** Pathlavath Shiva Kumar
- **Role:** Data Analysis Intern
- **Organization:** Unified Mentor Pvt. Ltd.

---

## Project Overview

This project focuses on **Predictive Modeling and Risk Scoring for Bank Customer Churn** using Machine Learning. The objective is to identify customers who are likely to leave the bank by assigning a churn probability score. This enables banks to proactively retain customers through targeted interventions.

The project is based on the **European Bank Customer Churn Dataset**, which contains information for approximately **10,000 customers** from **France, Germany, and Spain**, with an overall churn rate of nearly **20%**.

---

## Project Structure

```
bank_churn_project/
│
├── app/
│   └── app.py
│
├── data/
│   └── European_Bank.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   ├── feature_names.json
│   └── all_models.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── outputs/
│
├── reports/
│
├── src/
│   ├── data_prep.py
│   ├── train_model.py
│   └── explain.py
│
├── requirements.txt
└── README.md
```

---

# Project Features

- Data Cleaning and Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Machine Learning Model Training
- Model Performance Evaluation
- SHAP Explainability
- Interactive Streamlit Dashboard

---

# Machine Learning Models Used

The following models are trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

The best-performing model is selected based on the **ROC-AUC Score**.

---

# Feature Engineering

Several additional features were created to improve prediction performance.

Examples include:

- Balance / Salary Ratio
- Product Density
- Active Member × Number of Products
- Age × Tenure

---

# Explainable AI

The project uses **SHAP (SHapley Additive Explanations)** to explain model predictions.

The most influential features include:

- Number of Products
- Age
- Geography
- Balance
- Active Membership
- Balance Salary Ratio

---

# Streamlit Dashboard

The dashboard provides four interactive pages:

### 1. Churn Risk Calculator

Predicts customer churn probability based on user inputs.

### 2. Probability Distribution

Displays the distribution of churn probabilities across customers.

### 3. Feature Importance

Visualizes SHAP feature importance to explain model predictions.

### 4. What-If Simulator

Allows users to modify customer attributes and observe changes in predicted churn probability.

---

# Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

### Train Models

```bash
python src/train_model.py
```

This trains all machine learning models and saves the best-performing model inside the `models/` folder.

---

### Generate SHAP Explanations

```bash
python src/explain.py
```

This creates SHAP explanations and stores visualizations inside the `outputs/` folder.

---

### Launch Streamlit Dashboard

```bash
streamlit run app/app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

# Dataset

**Dataset Name:** European Bank Customer Churn Dataset

Number of Customers: **10,000**

Countries:

- France
- Germany
- Spain

Target Variable:

- Exited (Customer Churn)

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit

---

# Model Performance

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

The best model is automatically selected and saved for deployment.

---

# Future Improvements

- Hyperparameter Optimization
- Cross Validation
- Additional Feature Engineering
- Deep Learning Models
- Real-Time Customer Monitoring

---

# Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for customer churn prediction, including data preprocessing, feature engineering, predictive modeling, explainable AI using SHAP, and deployment through an interactive Streamlit dashboard. The solution enables proactive customer retention by identifying high-risk customers and providing interpretable predictions.

---

## Submitted By

**Pathlavath Shiva Kumar**

**Data Analysis Intern**

**Unified Mentor Pvt. Ltd.**

"""
Streamlit dashboard for the churn project. Four pages/modules:
  1. churn risk calculator - single customer, one prediction
  2. probability distribution - how the model's scores spread out
  3. feature importance - what's actually driving predictions
  4. what-if simulator - tweak a customer and see the risk move

Needs models/best_model.pkl etc to already exist (run train_model.py
and explain.py first). Launch with:
    streamlit run app/app.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# make sure this works whether streamlit is launched from the project
# root or from inside app/ - add project root to the path so `src` imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))  # so relative paths (data/, models/) resolve

st.set_page_config(page_title="Bank Churn Risk Dashboard", layout="wide")
with st.sidebar:
    st.markdown("## 👨‍💻 Student Details")
    st.write("**Name:** Pathlavath Shiva Kumar")
    st.write("**Role:** Data Analysis Intern")
    st.write("**Organization:** Unified Mentor Pvt. Ltd.")
    st.write("**Project:** Bank Churn Prediction")
    st.markdown("---")
    st.caption("Bank Churn Prediction Dashboard")
# ---------------------------------------------------------------
# load artifacts
# ---------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_names.json") as f:
        meta = json.load(f)
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, meta, metrics


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/European_Bank.csv")


model, scaler, meta, metrics = load_artifacts()
feature_names = meta["feature_names"]
numeric_cols = meta["numeric_cols"]
best_model_name = meta["best_model"]
raw_df = load_raw_data()


def build_feature_row(credit_score, geography, gender, age, tenure, balance,
                       num_products, has_cr_card, is_active, salary):
    """Turns raw inputs into the exact one-hot / engineered feature row
    the model was trained on."""
    row = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    row["BalanceSalaryRatio"] = row["Balance"] / row["EstimatedSalary"] if row["EstimatedSalary"] else 0
    row["ProductDensity"] = row["NumOfProducts"] / (row["Tenure"] + 1)
    row["EngagementProductScore"] = row["IsActiveMember"] * row["NumOfProducts"]
    row["AgeTenureInteraction"] = row["Age"] * row["Tenure"]

    df_row = pd.DataFrame([row])[feature_names]
    df_row[numeric_cols] = scaler.transform(df_row[numeric_cols])
    return df_row


def predict_prob(df_row):
    return model.predict_proba(df_row)[:, 1][0]


# ---------------------------------------------------------------
# sidebar navigation
# ---------------------------------------------------------------

st.sidebar.title("Bank Churn Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Churn Risk Calculator", "Probability Distribution",
     "Feature Importance", "What-If Simulator"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Best model: **{best_model_name}**")
st.sidebar.caption(f"Test ROC-AUC: **{metrics['results_all_models'][best_model_name]['roc_auc']:.3f}**")

# ---------------------------------------------------------------
# Overview
# ---------------------------------------------------------------

if page == "Overview":
    st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")
    st.write(
        "European Central Bank retail churn intelligence system. This dashboard "
        "turns the trained churn model into a set of tools a retention team can "
        "actually use day to day."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(raw_df):,}")
    col2.metric("Historical churn rate", f"{raw_df['Exited'].mean():.1%}")
    col3.metric("Best model", best_model_name)
    col4.metric("Test ROC-AUC", f"{metrics['results_all_models'][best_model_name]['roc_auc']:.3f}")

    st.subheader("Model comparison (held-out test set)")
    comp_df = pd.DataFrame(metrics["results_all_models"]).T
    comp_df = comp_df.rename(columns={
        "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1-Score", "roc_auc": "ROC-AUC"
    })
    st.dataframe(comp_df.style.format("{:.3f}").highlight_max(axis=0, color="#d4f4dd"))

    st.subheader("Confusion matrix - best model")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Retained", "Churned"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Retained", "Churned"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------------------------------------------------------------
# Module 1: Churn Risk Calculator
# ---------------------------------------------------------------

elif page == "Churn Risk Calculator":
    st.title("Customer Churn Risk Calculator")
    st.write("Enter a customer's profile to get a churn probability and risk flag.")

    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
    with c2:
        balance = st.number_input("Account Balance (€)", 0.0, 300000.0, 75000.0, step=1000.0)
        salary = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 100000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    with c3:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)

    threshold = st.slider("Risk threshold (churn flag cutoff)", 0.1, 0.9, 0.5, 0.05)

    if st.button("Calculate churn risk", type="primary"):
        row = build_feature_row(credit_score, geography, gender, age, tenure,
                                 balance, num_products, has_cr_card, is_active, salary)
        prob = predict_prob(row)
        flag = "HIGH RISK - likely to churn" if prob >= threshold else "LOW RISK - likely to stay"

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Churn probability", f"{prob:.1%}")
            if prob >= threshold:
                st.error(flag)
            else:
                st.success(flag)
        with col_b:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh([0], [1], color="#e8e8e8")
            ax.barh([0], [prob], color="#d9534f" if prob >= threshold else "#5cb85c")
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Churn probability")
            st.pyplot(fig)

        st.caption(
            "For high-risk customers it's probably worth a retention offer or "
            "just a check-in call, especially if they're not an active member "
            "or they're holding 3+ products (that combo shows up a lot in the churned group)."
        )

# ---------------------------------------------------------------
# Module 2: Probability Distribution Visualization
# ---------------------------------------------------------------

elif page == "Probability Distribution":
    st.title("Churn Probability Distribution")
    st.write(
        "Distribution of predicted churn probabilities across the customer "
        "base, split by actual churn outcome. Useful for picking a sensible "
        "risk threshold for retention campaigns."
    )

    from src.data_prep import build_dataset, scale_numeric  # noqa
    from sklearn.model_selection import train_test_split

    X, y, _ = build_dataset("data/European_Bank.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    _, X_test_scaled, _, _ = scale_numeric(X_train, X_test, numeric_cols)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(probs[y_test.values == 0], bins=30, alpha=0.6, label="Actually retained", color="#5cb85c")
    ax.hist(probs[y_test.values == 1], bins=30, alpha=0.6, label="Actually churned", color="#d9534f")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Number of customers")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Risk segments")
    seg_df = pd.DataFrame({"probability": probs, "actual": y_test.values})
    seg_df["segment"] = pd.cut(
        seg_df["probability"], bins=[0, 0.3, 0.6, 1.0],
        labels=["Low risk (<30%)", "Medium risk (30-60%)", "High risk (>60%)"]
    )
    summary = seg_df.groupby("segment").agg(
        customers=("actual", "size"),
        actual_churn_rate=("actual", "mean"),
    ).round(3)
    st.dataframe(summary)

# ---------------------------------------------------------------
# Module 3: Feature Importance Dashboard
# ---------------------------------------------------------------

elif page == "Feature Importance":
    st.title("Feature Importance Dashboard")

    tab1, tab2 = st.tabs(["Model feature importance", "SHAP values"])

    with tab1:
        imp = pd.Series(metrics["feature_importance"]).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.tail(12).plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top drivers - {best_model_name}")
        st.pyplot(fig)

    with tab2:
        try:
            with open("models/shap_summary.json") as f:
                shap_imp = json.load(f)
            imp2 = pd.Series(shap_imp).sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            imp2.tail(12).plot(kind="barh", ax=ax2, color="#C44E52")
            ax2.set_xlabel("Mean |SHAP value|")
            ax2.set_title("Top drivers - SHAP")
            st.pyplot(fig2)
            st.caption("Run `python src/explain.py` to regenerate these numbers if the model changes.")
        except FileNotFoundError:
            st.info("Run `python src/explain.py` first to generate SHAP values.")

    st.subheader("What this means")
    st.markdown(
        "- **Age** and **Number of Products** are consistently the strongest "
        "drivers of churn risk.\n"
        "- **Geography (Germany)** and the **engagement x product** interaction "
        "add meaningful lift on top of the raw features.\n"
        "- Activity status matters more than credit card ownership - engagement, "
        "not paperwork, drives retention."
    )

# ---------------------------------------------------------------
# Module 4: What-If Scenario Simulator
# ---------------------------------------------------------------

elif page == "What-If Simulator":
    st.title("What-If Scenario Simulator")
    st.write(
        "Pick a customer profile as a baseline, then adjust engagement / "
        "product variables to see how churn probability responds. This is "
        "meant to support conversations like *'if we get this customer to "
        "become an active member, how much does their risk drop?'*"
    )

    st.subheader("Baseline customer")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score ", 300, 900, 600, key="ws_cs")
        age = st.slider("Age ", 18, 92, 45, key="ws_age")
        tenure = st.slider("Tenure ", 0, 10, 3, key="ws_ten")
    with c2:
        balance = st.number_input("Balance (€) ", 0.0, 300000.0, 50000.0, step=1000.0, key="ws_bal")
        salary = st.number_input("Salary (€) ", 0.0, 250000.0, 90000.0, step=1000.0, key="ws_sal")
        geography = st.selectbox("Geography ", ["France", "Germany", "Spain"], key="ws_geo")
    with c3:
        gender = st.selectbox("Gender ", ["Female", "Male"], key="ws_gen")
        has_cr_card = st.checkbox("Has Credit Card ", value=True, key="ws_card")

    st.subheader("Scenario levers")
    l1, l2 = st.columns(2)
    with l1:
        base_active = st.radio("Baseline: active member?", ["No", "Yes"], horizontal=True, key="ws_base_active")
        base_products = st.select_slider("Baseline: number of products", [1, 2, 3, 4], value=1, key="ws_base_prod")
    with l2:
        scenario_active = st.radio("Scenario: active member?", ["No", "Yes"], horizontal=True, index=1, key="ws_scn_active")
        scenario_products = st.select_slider("Scenario: number of products", [1, 2, 3, 4], value=2, key="ws_scn_prod")

    if st.button("Run comparison", type="primary"):
        base_row = build_feature_row(credit_score, geography, gender, age, tenure, balance,
                                      base_products, has_cr_card, base_active == "Yes", salary)
        scenario_row = build_feature_row(credit_score, geography, gender, age, tenure, balance,
                                          scenario_products, has_cr_card, scenario_active == "Yes", salary)

        base_prob = predict_prob(base_row)
        scenario_prob = predict_prob(scenario_row)
        delta = scenario_prob - base_prob

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline churn probability", f"{base_prob:.1%}")
        c2.metric("Scenario churn probability", f"{scenario_prob:.1%}", delta=f"{delta:+.1%}")
        c3.metric("Risk change", "Lower risk" if delta < 0 else ("Higher risk" if delta > 0 else "No change"))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(["Baseline", "Scenario"], [base_prob, scenario_prob],
               color=["#999999", "#5cb85c" if delta < 0 else "#d9534f"])
        ax.set_ylabel("Churn probability")
        ax.set_ylim(0, 1)
        for i, v in enumerate([base_prob, scenario_prob]):
            ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
        st.pyplot(fig)

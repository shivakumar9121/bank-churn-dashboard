"""
Cleans up the raw csv and turns it into something a model can train on.
Kept separate from train_model.py so the notebook and the streamlit app
can both reuse the same logic instead of copy-pasting it everywhere.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/European_Bank.csv"

# these don't carry any signal, get rid of them
DROP_COLS = ["CustomerId", "Surname", "Year"]

NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "BalanceSalaryRatio", "ProductDensity", "EngagementProductScore",
    "AgeTenureInteraction",
]


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def add_engineered_features(df):
    """The 4 extra features called out in the brief."""
    df = df.copy()

    # balance to salary ratio - guard against divide by zero even though
    # salary is never actually 0 in this dataset
    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].replace(0, np.nan)
    df["BalanceSalaryRatio"] = df["BalanceSalaryRatio"].fillna(0)

    # products relative to how long they've been a customer - a longer
    # tenure customer with only 1 product looks different than someone
    # who's had 4 products in year 1
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # this one turned out to matter more than I expected - active member
    # x number of products
    df["EngagementProductScore"] = df["IsActiveMember"] * df["NumOfProducts"]

    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

    return df


def clean_and_encode(df, drop_target=False):
    """
    Full pipeline - drop id columns, add engineered features, one hot
    encode geography/gender. Returns X, y separately.
    """
    df = df.copy()

    # no missing values in this dataset but leaving this in just in case
    # a future export has gaps somewhere
    df = df.dropna(subset=["CreditScore", "Age", "Balance", "EstimatedSalary"])

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = add_engineered_features(df)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    y = df["Exited"] if "Exited" in df.columns else None
    X = df.drop(columns=["Exited"]) if "Exited" in df.columns else df

    return X, y


def scale_numeric(X_train, X_test, numeric_cols=None):
    """Fit scaler on train only, obviously, then apply to both."""
    if numeric_cols is None:
        numeric_cols = [c for c in NUMERIC_COLS if c in X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler, numeric_cols


def build_dataset(path=RAW_PATH):
    """One call that does load + clean, used by train_model.py."""
    raw = load_raw(path)
    X, y = clean_and_encode(raw)
    return X, y, raw

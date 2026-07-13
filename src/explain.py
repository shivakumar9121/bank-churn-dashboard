"""
SHAP explainability for whatever model train_model.py picked as the
winner. Saves the plots to outputs/ (used these directly in the report)
and dumps a summary json for the streamlit app to read.

Needs models/best_model.pkl to already exist, so run train_model.py
first:
    python src/train_model.py
    python src/explain.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from data_prep import build_dataset, scale_numeric

RANDOM_STATE = 42


def main():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    with open("models/feature_names.json") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    numeric_cols = meta["numeric_cols"]

    X, y, raw = build_dataset("data/European_Bank.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train_scaled, X_test_scaled, _, _ = scale_numeric(X_train, X_test, numeric_cols)

    # SHAP on the full test set is slow and 1000 rows is plenty to get a
    # stable-looking summary plot
    sample = X_test_scaled.sample(n=min(1000, len(X_test_scaled)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # some sklearn classifiers return a list of arrays (one per class),
    # others just return one array for binary classification - handle both
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = dict(sorted(
        zip(feature_names, mean_abs_shap.tolist()), key=lambda x: -x[1]
    ))

    with open("models/shap_summary.json", "w") as f:
        json.dump(shap_importance, f, indent=2)

    # --- plots for the report ---
    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary_plot.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_bar_plot.png", dpi=150)
    plt.close()

    # quick partial dependence view for whatever feature came out on top -
    # sweep it across a range of realistic values and see how the average
    # predicted probability moves, holding everything else where it is
    top_feature = list(shap_importance.keys())[0]
    if top_feature in numeric_cols:
        grid = np.linspace(X_test[top_feature].quantile(0.02), X_test[top_feature].quantile(0.98), 30)
        avg_probs = []
        base = X_test_scaled.copy()
        col_mean = X_train[top_feature].mean()
        col_std = X_train[top_feature].std()
        for val in grid:
            scaled_val = (val - col_mean) / col_std
            temp = base.copy()
            temp[top_feature] = scaled_val
            avg_probs.append(model.predict_proba(temp)[:, 1].mean())

        plt.figure(figsize=(7, 4.5))
        plt.plot(grid, avg_probs, marker="o", markersize=3)
        plt.xlabel(top_feature)
        plt.ylabel("Average predicted churn probability")
        plt.title(f"Partial dependence: churn probability vs {top_feature}")
        plt.tight_layout()
        plt.savefig("outputs/partial_dependence_top_feature.png", dpi=150)
        plt.close()

    print("Top SHAP drivers:")
    for k, v in list(shap_importance.items())[:8]:
        print(f"  {k}: {v:.4f}")
    print("\nSaved plots to outputs/, summary to models/shap_summary.json")


if __name__ == "__main__":
    main()

"""
Trains all 5 models from the brief (Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting, XGBoost), evaluates them on a held-out
test set, and saves whichever one has the best ROC-AUC so the app can
load it later.

Run this from the project root:
    python src/train_model.py

Takes maybe 20-30 seconds on my laptop, most of it is the Random Forest
and XGBoost fitting.
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

from data_prep import build_dataset, scale_numeric

RANDOM_STATE = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, scale_pos_weight=3.9  # roughly majority/minority ratio, 79.6/20.4
        ),
    }


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }, probs, preds


def main():
    print("Loading and preparing data...")
    X, y, raw = build_dataset("data/European_Bank.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    X_train_scaled, X_test_scaled, scaler, numeric_cols = scale_numeric(X_train, X_test)

    feature_names = list(X_train_scaled.columns)

    results = {}
    trained_models = {}
    roc_curves = {}

    for name, model in get_models().items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        metrics, probs, preds = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics
        trained_models[name] = model

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        print(f"  accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # need the confusion matrix for the report too
    best_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    best_preds = (best_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_preds).tolist()

    # tree models have .feature_importances_, logistic regression doesn't -
    # fall back to abs(coef_) for that case
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, "coef_"):
        importances = dict(zip(feature_names, np.abs(best_model.coef_[0]).tolist()))
    else:
        importances = {}
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    # --- save everything the app / report needs ---
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names, "models/feature_names.json.pkl")

    with open("models/feature_names.json", "w") as f:
        json.dump({"feature_names": feature_names, "numeric_cols": numeric_cols, "best_model": best_name}, f, indent=2)

    with open("models/metrics.json", "w") as f:
        json.dump({
            "results_all_models": results,
            "best_model": best_name,
            "confusion_matrix": cm,
            "feature_importance": importances,
            "roc_curves": roc_curves,
            "test_set_size": len(y_test),
            "churn_rate_test": float(y_test.mean()),
        }, f, indent=2)

    # also dump all trained models in case app wants to compare / switch
    joblib.dump(trained_models, "models/all_models.pkl")

    print("\nSaved: models/best_model.pkl, models/scaler.pkl, models/feature_names.json, models/metrics.json")


if __name__ == "__main__":
    main()

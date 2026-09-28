"""
Employee Income Prediction — model training script.

Trains and compares Logistic Regression, Random Forest, and Gradient
Boosting on the UCI Adult / Census Income dataset, then saves the
best-performing model (plus the label encoders it needs at inference
time) to disk for the Streamlit app (app.py) to load.

Usage:
    python train_model.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "adult.csv"
MODEL_PATH = "model.pkl"
ENCODERS_PATH = "encoders.pkl"
RESULTS_PATH = "results.json"

TARGET_COL = "income"
DROP_COLS = ["fnlwgt"]  # census sampling weight — no predictive signal


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace("?", np.nan).dropna()
    return df


def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df):,} clean records after dropping missing values")

    y = (df[TARGET_COL] == ">50K").astype(int)
    X = df.drop(columns=[TARGET_COL, *DROP_COLS])

    # Label-encode every categorical column and keep the encoders —
    # the app needs the exact same mapping at prediction time.
    encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include=["object", "str"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = round(acc * 100, 2)
        fitted[name] = model
        print(f"  accuracy = {acc * 100:.1f}%")

    best_name = max(results, key=results.get)
    best_model = fitted[best_name]
    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)

    print(f"\nBest model: {best_name} ({results[best_name]:.1f}% accuracy)")

    joblib.dump(
        {"model": best_model, "model_name": best_name, "feature_order": list(X.columns)},
        MODEL_PATH,
    )
    joblib.dump(encoders, ENCODERS_PATH)

    summary = {
        "results": results,
        "best_name": best_name,
        "cm": cm.tolist(),
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importance": (
            dict(zip(X.columns, best_model.feature_importances_.round(4)))
            if hasattr(best_model, "feature_importances_")
            else {}
        ),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {MODEL_PATH}, {ENCODERS_PATH}, {RESULTS_PATH}")


if __name__ == "__main__":
    main()

"""
train_model.py

Phase 7: Binary human-presence classifier (Human vs Empty Room).
Phase 8: Multi-class activity classifier (empty / sitting / standing / walking).

Trains several algorithms per the roadmap, evaluates with accuracy/precision/
recall/F1, picks the best model per task, and saves it + metrics with joblib
for the Flask dashboard (app.py) to load.

Run:
    python data/generate_synthetic_csi.py   # first, to create the CSV
    python train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from features import featurize_dataframe

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "data", "synthetic_csi_dataset.csv")
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

WINDOW_LEN = 100
N_SUBCARRIERS = 30


def evaluate(name, model, X_test, y_test, average="binary"):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average=average, zero_division=0),
        "recall": recall_score(y_test, preds, average=average, zero_division=0),
        "f1": f1_score(y_test, preds, average=average, zero_division=0),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    return metrics


def run_presence_detection(X_train, X_test, y_train_bin, y_test_bin, scaler):
    """Phase 7: Human present (1) vs No Human / empty room (0)."""
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train_bin)
        m = evaluate(name, model, X_test, y_test_bin, average="binary")
        results[name] = m
        if m["f1"] > best_f1:
            best_name, best_model, best_f1 = name, model, m["f1"]

    print(f"\nBest presence-detection model: {best_name} (F1={best_f1:.3f})")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "presence_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "presence_scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "presence_metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    return best_name, results


def run_activity_recognition(X_train, X_test, y_train, y_test, scaler, class_names):
    """Phase 8: multi-class activity recognition."""
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, eval_metric="mlogloss", random_state=42
        )
    else:
        print("\n[note] xgboost not installed — skipping XGBoost, using tree/forest/SVM only.")

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        m = evaluate(name, model, X_test, y_test, average="macro")
        results[name] = m
        if m["f1"] > best_f1:
            best_name, best_model, best_f1 = name, model, m["f1"]

    print(f"\nBest activity-recognition model: {best_name} (macro F1={best_f1:.3f})")
    print("\nDetailed report for best model:")
    print(classification_report(y_test, best_model.predict(X_test), target_names=class_names))

    joblib.dump(best_model, os.path.join(MODELS_DIR, "activity_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "activity_scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "activity_metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results, "classes": list(class_names)}, f, indent=2)
    return best_name, results


def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Run `python data/generate_synthetic_csi.py` first."
        )

    raw_df = pd.read_csv(DATA_CSV)
    X, y, activities = featurize_dataframe(raw_df, WINDOW_LEN, N_SUBCARRIERS)

    class_names = sorted(raw_df[["activity", "label"]].drop_duplicates()["activity"].tolist(),
                          key=lambda a: raw_df[raw_df.activity == a]["label"].iloc[0])

    # ---- Phase 7: binary presence label (empty=0 vs everything else=1) ----
    y_bin = (raw_df["activity"] != "empty").astype(int).values

    X_train, X_test, y_train, y_test, ybin_train, ybin_test = train_test_split(
        X, y, y_bin, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("=" * 60)
    print("PHASE 7: Human Presence Detection (binary)")
    print("=" * 60)
    run_presence_detection(X_train_s, X_test_s, ybin_train, ybin_test, scaler)

    print("\n" + "=" * 60)
    print("PHASE 8: Activity Recognition (multi-class)")
    print("=" * 60)
    scaler2 = StandardScaler()
    X_train_s2 = scaler2.fit_transform(X_train)
    X_test_s2 = scaler2.transform(X_test)
    run_activity_recognition(X_train_s2, X_test_s2, y_train, y_test, scaler2, class_names)

    # Save feature column order — app.py needs this to build feature vectors identically
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)

    print("\nAll models + scalers + metrics saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()

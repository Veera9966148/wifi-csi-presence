"""
app.py

Phase 9: Dashboard backend (Flask). Phase 10 target: deploy this on
Render/Railway (see README for steps).

Because there is no real CSI-capturing NIC attached to this environment,
/api/predict simulates one incoming CSI window per request using the same
generator as training (data/generate_synthetic_csi.py) and runs it through
the trained presence + activity models. Swap `simulate_incoming_csi_window()`
for a real CSI reader (e.g. parsing Intel 5300 CSI tool output or an ESP32
CSI stream) to go from prototype to a real deployment — nothing else in this
file needs to change, since the feature extraction and model interface stay
identical.
"""

import os
import json
import random
import numpy as np
import joblib
from flask import Flask, jsonify, render_template

from features import extract_features_from_window
from generate_synthetic_csi import generate_window, CLASSES, WINDOW_LEN, N_SUBCARRIERS

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")

app = Flask(__name__)

# ---- Load trained artifacts once at startup ----
presence_model = joblib.load(os.path.join(MODELS_DIR, "presence_model.joblib"))
presence_scaler = joblib.load(os.path.join(MODELS_DIR, "presence_scaler.joblib"))
activity_model = joblib.load(os.path.join(MODELS_DIR, "activity_model.joblib"))
activity_scaler = joblib.load(os.path.join(MODELS_DIR, "activity_scaler.joblib"))

with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)

with open(os.path.join(MODELS_DIR, "activity_metrics.json")) as f:
    ACTIVITY_METRICS = json.load(f)

with open(os.path.join(MODELS_DIR, "presence_metrics.json")) as f:
    PRESENCE_METRICS = json.load(f)


def simulate_incoming_csi_window():
    """
    Stand-in for a real CSI capture. Replace this function with code that
    reads the latest WINDOW_LEN x N_SUBCARRIERS amplitude window from your
    actual Wi-Fi CSI hardware/log stream. Everything downstream (feature
    extraction, scaling, model.predict) is hardware-agnostic.
    """
    activity = random.choice(CLASSES)
    return generate_window(activity), activity  # ground-truth returned only for demo labeling


def featurize_single_window(window: np.ndarray) -> np.ndarray:
    feats = extract_features_from_window(window)
    # Order features exactly as during training
    vec = np.array([[feats[col] for col in FEATURE_COLUMNS]])
    return vec


@app.route("/")
def index():
    return render_template(
        "index.html",
        activity_metrics=ACTIVITY_METRICS,
        presence_metrics=PRESENCE_METRICS,
    )


@app.route("/api/predict")
def predict():
    window, true_activity = simulate_incoming_csi_window()
    feat_vec = featurize_single_window(window)

    presence_feat_scaled = presence_scaler.transform(feat_vec)
    presence_pred = int(presence_model.predict(presence_feat_scaled)[0])
    presence_proba = float(np.max(presence_model.predict_proba(presence_feat_scaled)))

    activity_feat_scaled = activity_scaler.transform(feat_vec)
    activity_pred_idx = int(activity_model.predict(activity_feat_scaled)[0])
    activity_proba = float(np.max(activity_model.predict_proba(activity_feat_scaled)))
    activity_name = CLASSES[activity_pred_idx]

    return jsonify({
        "human_detected": bool(presence_pred),
        "presence_confidence": round(presence_proba * 100, 1),
        "activity": activity_name,
        "activity_confidence": round(activity_proba * 100, 1),
        "simulated_ground_truth": true_activity,  # remove in a real deployment
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

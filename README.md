# AI-Powered Contactless Human Presence Detection Using Wi-Fi CSI

## Problem Statement
Detect whether a human is present in a room, and what they're doing (sitting,
standing, walking), using only Wi-Fi Channel State Information (CSI) — no
cameras, no wearables, no motion PIR sensors. CSI describes how a Wi-Fi signal
is distorted by multipath propagation as it bounces off walls, furniture, and
bodies; a person's presence and movement measurably perturbs that distortion
pattern, which is what the model learns to recognize.

## Objective
Build and evaluate a full pipeline — data → features → classical ML models →
live dashboard — for (1) binary human presence detection and (2) multi-class
activity recognition (empty / sitting / standing / walking), and package it
as a demonstrable, deployable prototype.

## ⚠️ Status: prototype on synthetic data — read before reusing any number
No CSI hardware or downloaded real dataset (Widar 3.0 / SignFi / CSI-HAR /
Fall-Detection-CSI) was available in this build environment, so
`data/generate_synthetic_csi.py` generates *physically-motivated but
synthetic* CSI windows: stable baseline for an empty room, low-amplitude
quasi-periodic fluctuation for sitting/standing (mimicking breathing-rate
micro-motion), and larger broadband fluctuation for walking (mimicking
multipath fading from body motion). This is enough to validate the entire
pipeline end to end, but **the accuracy numbers below describe how well the
models separate this synthetic generator's own classes — not real-world
sensing accuracy.** Do not put these numbers on a resume as if they came from
real CSI. Swap `data/synthetic_csi_dataset.csv` for a real dataset (same
column format: flattened CSI window + activity + label) and rerun
`train_model.py` before trusting any figure.

## Dataset
- **Current**: synthetic, 1,200 samples (300 per class), 100 timesteps × 30
  subcarriers per window (~2 s @ 50 Hz, matching common Intel 5300 NIC CSI
  captures).
- **To go real**: download Widar 3.0, SignFi, or a CSI-HAR dataset, reshape
  each capture into the same `(window, subcarriers) → flatten → label` CSV
  format `features.py` expects, and everything downstream is unchanged.

## Pipeline
1. `data/generate_synthetic_csi.py` — generates the CSI dataset (swap for
   real data loader).
2. `features.py` — extracts 12 statistical/frequency features per window
   (mean, std, variance, skew, kurtosis, dominant FFT frequency/magnitude,
   frame-to-frame motion energy, subcarrier spread).
3. `train_model.py` — trains Logistic Regression, Decision Tree, Random
   Forest, SVM, and XGBoost; evaluates accuracy/precision/recall/F1; keeps
   the best model per task.
4. `app.py` + `templates/index.html` — Flask dashboard showing live
   presence status, confidence, and activity, polling a `/api/predict`
   endpoint every 2 seconds.

## Results (synthetic data — see warning above)
| Task | Best model | Accuracy | F1 |
|---|---|---|---|
| Presence detection (binary) | Logistic Regression | 100.0% | 1.000 |
| Activity recognition (4-class) | Random Forest | 86.7% | 0.867 (macro) |

Presence detection is trivially easy here because "empty" vs. "any human
signature" is a large, clean gap in the synthetic generator. Activity
recognition is harder and shows real confusion between sitting and standing
(both are low-amplitude, low-frequency micro-motion classes) — this pattern
is consistent with what's reported in real CSI literature, which is a good
sign the synthetic generator is qualitatively realistic, not proof the model
will hit 86.7% on real data.

## How to run
```bash
pip install numpy pandas scikit-learn scipy joblib flask xgboost

python data/generate_synthetic_csi.py   # builds data/synthetic_csi_dataset.csv
python train_model.py                   # trains + saves models/*.joblib
python app.py                           # dashboard at http://localhost:5000
```

## Deployment (Phase 10)
Push this repo to GitHub, then deploy on Render or Railway as a standard
Flask app:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app` (add `gunicorn` to requirements for
  production — the built-in Flask dev server used for local testing is not
  meant for production traffic)
- Ensure `models/*.joblib` and `models/*.json` are committed (or regenerated
  by a build-step call to `train_model.py`) so the deployed app has trained
  artifacts to load.

## Future scope
- Replace the synthetic generator with a real CSI capture pipeline (Intel
  5300 CSI tool, Atheros CSI tool, or an ESP32 CSI firmware) — `app.py`'s
  `simulate_incoming_csi_window()` is the one function to swap.
- Add CNN/LSTM models on raw windows instead of hand-engineered features,
  once real labeled data volume justifies it.
- Multi-room / multi-AP fusion for whole-home coverage.
- On-device / edge deployment for privacy (no CSI leaves the router).

## Resume-ready summary
**AI-Powered Human Presence Detection using Wi-Fi CSI** — Python, scikit-learn,
XGBoost, Flask, NumPy, pandas. Built and evaluated a full presence-detection
and activity-recognition pipeline (feature engineering → classical ML →
live dashboard) as a prototype; validated the architecture on synthetic
CSI signatures pending access to real hardware/dataset. *(Only claim real
accuracy figures after retraining on an actual CSI dataset.)*

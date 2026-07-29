"""
features.py

Turns a raw CSI window (WINDOW_LEN timesteps x N_SUBCARRIERS) into a compact
feature vector for classical ML models (Logistic Regression, Random Forest,
SVM, XGBoost). This is Phase 5 (cleaning/feature extraction) + part of
Phase 7 in the roadmap, made concrete and runnable.

Features per window, aggregated across subcarriers:
  - mean, std, variance          (signal level + spread)
  - min, max, peak-to-peak       (range of fluctuation)
  - skewness, kurtosis           (distribution shape -> motion vs static)
  - dominant FFT frequency       (periodicity, e.g. breathing ~0.2-0.4 Hz)
  - dominant FFT magnitude       (strength of that periodicity)
  - mean absolute difference     (frame-to-frame variation -> motion energy)
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

SAMPLING_RATE_HZ = 50.0


def extract_features_from_window(window: np.ndarray) -> dict:
    """window: shape (T, C) raw CSI amplitude values."""
    # Collapse across subcarriers to a single amplitude time series (mean over C)
    ts = window.mean(axis=1)  # shape (T,)

    diffs = np.diff(ts)
    fft_vals = np.abs(rfft(ts - ts.mean()))
    fft_freqs = rfftfreq(len(ts), d=1.0 / SAMPLING_RATE_HZ)

    # Ignore DC component (index 0) when finding dominant frequency
    if len(fft_vals) > 1:
        dom_idx = np.argmax(fft_vals[1:]) + 1
        dom_freq = fft_freqs[dom_idx]
        dom_mag = fft_vals[dom_idx]
    else:
        dom_freq, dom_mag = 0.0, 0.0

    feats = {
        "mean": ts.mean(),
        "std": ts.std(),
        "variance": ts.var(),
        "min": ts.min(),
        "max": ts.max(),
        "peak_to_peak": ts.max() - ts.min(),
        "skewness": skew(ts),
        "kurtosis": kurtosis(ts),
        "dominant_freq_hz": dom_freq,
        "dominant_freq_mag": dom_mag,
        "mean_abs_diff": np.mean(np.abs(diffs)),
        "subcarrier_std_mean": window.std(axis=0).mean(),  # spread across subcarriers
    }
    return feats


def featurize_dataframe(raw_df, window_len: int, n_subcarriers: int):
    """
    raw_df: DataFrame produced by generate_synthetic_csi.build_dataset(),
            with WINDOW_LEN*N_SUBCARRIERS flattened columns + activity/label.
    Returns: (X feature DataFrame, y labels array, activity name array)
    """
    import pandas as pd

    feature_rows = []
    sc_cols = [c for c in raw_df.columns if c.startswith("sc")]

    for _, row in raw_df.iterrows():
        flat = row[sc_cols].values.astype(float)
        window = flat.reshape(window_len, n_subcarriers)
        feats = extract_features_from_window(window)
        feature_rows.append(feats)

    X = pd.DataFrame(feature_rows)
    y = raw_df["label"].values
    activities = raw_df["activity"].values
    return X, y, activities

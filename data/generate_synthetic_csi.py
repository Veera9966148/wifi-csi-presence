"""
generate_synthetic_csi.py

Generates a synthetic Wi-Fi CSI dataset for prototyping the presence/activity
detection pipeline BEFORE real hardware/dataset access is available.

WHY SYNTHETIC DATA (read this before you trust any accuracy number):
Real CSI amplitude traces show class-dependent statistical signatures:
  - Empty room: low variance, no periodic component, amplitude ~stationary.
  - Standing/Sitting (micro-motion, e.g. breathing): low-amplitude, low-frequency
    periodicity (~0.2-0.5 Hz breathing rate) riding on a stable baseline.
  - Walking (macro-motion): larger-amplitude, broadband fluctuations from
    multipath fading as the body moves through the Fresnel zones.
This generator reproduces those *qualitative* signatures with tunable noise,
NOT real physical CSI. Do not report model accuracy trained on this data as
if it reflects real-world performance. Replace this module's output with a
real dataset (Widar 3.0, SignFi, CSI-HAR, etc.) before drawing conclusions.

Each sample = one time window of CSI amplitude readings across 30 subcarriers
(a common CSI dimension for Intel 5300 NIC captures), summarized as a single
row of (n_subcarriers * n_timesteps) raw values + label. Downstream feature
extraction (features.py) turns each window into a compact feature vector.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

N_SUBCARRIERS = 30
WINDOW_LEN = 100          # timesteps per window (~2 sec @ 50 Hz sampling)
SAMPLES_PER_CLASS = 300

CLASSES = ["empty", "sitting", "standing", "walking"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}


def _baseline_amplitude(n_subcarriers):
    """Static multipath baseline per subcarrier (differs by frequency)."""
    return 20 + 5 * np.sin(np.linspace(0, 3, n_subcarriers)) + RNG.normal(0, 0.5, n_subcarriers)


def generate_window(activity: str) -> np.ndarray:
    """Generate one (WINDOW_LEN, N_SUBCARRIERS) CSI amplitude window for a class."""
    base = _baseline_amplitude(N_SUBCARRIERS)
    t = np.linspace(0, WINDOW_LEN / 50.0, WINDOW_LEN)  # seconds, 50 Hz sampling

    window = np.tile(base, (WINDOW_LEN, 1))  # (T, C)

    if activity == "empty":
        noise_std = 0.15
        window += RNG.normal(0, noise_std, window.shape)

    elif activity in ("sitting", "standing"):
        # Small quasi-periodic breathing-rate fluctuation + sensor noise
        breathing_freq = RNG.uniform(0.2, 0.4)
        amp = RNG.uniform(0.4, 0.9) if activity == "sitting" else RNG.uniform(0.6, 1.2)
        phase_per_subcarrier = RNG.uniform(0, 2 * np.pi, N_SUBCARRIERS)
        signal = amp * np.sin(2 * np.pi * breathing_freq * t)[:, None] + \
                 0.2 * np.sin(phase_per_subcarrier)[None, :]
        window += signal + RNG.normal(0, 0.3, window.shape)

    elif activity == "walking":
        # Broadband, larger-amplitude fluctuations from body motion / multipath fading
        n_components = 4
        signal = np.zeros((WINDOW_LEN, N_SUBCARRIERS))
        for _ in range(n_components):
            freq = RNG.uniform(0.5, 2.5)
            amp = RNG.uniform(1.5, 4.0)
            phase = RNG.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase)[:, None]
        window += signal + RNG.normal(0, 0.8, window.shape)

    else:
        raise ValueError(f"Unknown activity: {activity}")

    return np.clip(window, 0, None)


def build_dataset() -> pd.DataFrame:
    rows = []
    for activity in CLASSES:
        for _ in range(SAMPLES_PER_CLASS):
            window = generate_window(activity)  # (T, C)
            flat = window.flatten()             # length T*C
            row = list(flat) + [activity, LABEL_MAP[activity]]
            rows.append(row)

    col_names = [f"sc{c}_t{t}" for t in range(WINDOW_LEN) for c in range(N_SUBCARRIERS)]
    col_names += ["activity", "label"]
    df = pd.DataFrame(rows, columns=col_names)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    df = build_dataset()
    out_path = os.path.join(out_dir, "synthetic_csi_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} samples -> {out_path}")
    print(df["activity"].value_counts())

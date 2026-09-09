"""Evaluate temporal anomaly detection across multiple thresholds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.anomaly.inject_anomalies import (
    inject_calibration_drift,
    inject_frozen_temperature,
    inject_humidity_spike,
    inject_multivariate_inconsistency,
    inject_pressure_spike,
    inject_temperature_spike,
)
from src.anomaly.temporal_detector import TemporalAnomalyDetector
from src.models.sequence_data import prepare_lstm_data


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imd_maitri_member1_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lstm_autoencoder.pt"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "lstm_scaler.joblib"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEQUENCE_LENGTH = 24
N_NORMAL = 500

THRESHOLDS = [
    0.05,
    0.07,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25,
]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

data = prepare_lstm_data(
    str(CSV_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


# ---------------------------------------------------------
# Convert evaluation data back to physical values
# ---------------------------------------------------------

n_samples, n_steps, n_features = data.X_eval.shape

raw_eval = data.scaler.inverse_transform(
    data.X_eval.reshape(-1, n_features)
).reshape(
    n_samples,
    n_steps,
    n_features,
)

raw_eval = raw_eval[:N_NORMAL]


# ---------------------------------------------------------
# Create detector
# ---------------------------------------------------------

detector = TemporalAnomalyDetector(
    model_path=str(MODEL_PATH),
    scaler_path=str(SCALER_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


# ---------------------------------------------------------
# Build normal and anomalous datasets
# ---------------------------------------------------------

normal = raw_eval.copy()

temperature_spikes = np.array(
    [
        inject_temperature_spike(sequence, magnitude=20.0)
        for sequence in raw_eval
    ],
    dtype=np.float32,
)

pressure_spikes = np.array(
    [
        inject_pressure_spike(sequence, magnitude=40.0)
        for sequence in raw_eval
    ],
    dtype=np.float32,
)

humidity_spikes = np.array(
    [
        inject_humidity_spike(sequence, value=100.0)
        for sequence in raw_eval
    ],
    dtype=np.float32,
)

frozen_sensors = np.array(
    [
        inject_frozen_temperature(sequence, duration=8)
        for sequence in raw_eval
    ],
    dtype=np.float32,
)

calibration_drifts = np.array(
    [
        inject_calibration_drift(
            sequence,
            drift_per_step=0.5,
        )
        for sequence in raw_eval
    ],
    dtype=np.float32,
)

multivariate = np.array(
    [
        inject_multivariate_inconsistency(
            sequence,
            temperature_offset=15.0,
        )
        for sequence in raw_eval
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------
# Combine datasets
# ---------------------------------------------------------

anomalies = np.concatenate(
    [
        temperature_spikes,
        pressure_spikes,
        humidity_spikes,
        frozen_sensors,
        calibration_drifts,
        multivariate,
    ],
    axis=0,
)

X = np.concatenate(
    [
        normal,
        anomalies,
    ],
    axis=0,
)


# Labels:
# 0 = normal
# 1 = anomaly

y_true = np.concatenate(
    [
        np.zeros(len(normal), dtype=int),
        np.ones(len(anomalies), dtype=int),
    ]
)


# ---------------------------------------------------------
# Calculate reconstruction errors
# ---------------------------------------------------------

print("\nCalculating reconstruction errors...\n")

errors = detector.calculate_reconstruction_error(
    X,
    already_scaled=False,
)


# ---------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------

print("=" * 90)
print("TEMPORAL LSTM THRESHOLD EVALUATION")
print("=" * 90)

print(
    f"\nNormal sequences : {len(normal)}"
)

print(
    f"Anomaly sequences: {len(anomalies)}"
)

print(
    f"Total sequences  : {len(X)}"
)

print("\n")


print(
    f"{'Threshold':<12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'FPR':>12}"
    f"{'TP':>8}"
    f"{'FP':>8}"
    f"{'FN':>8}"
    f"{'TN':>8}"
)

print("-" * 90)


for threshold in THRESHOLDS:

    y_pred = (
        errors > threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    false_positive_rate = fp / (
        fp + tn
    )

    print(
        f"{threshold:<12.3f}"
        f"{precision:>12.4f}"
        f"{recall:>12.4f}"
        f"{f1:>12.4f}"
        f"{false_positive_rate:>12.4f}"
        f"{tp:>8}"
        f"{fp:>8}"
        f"{fn:>8}"
        f"{tn:>8}"
    )


print("=" * 90)
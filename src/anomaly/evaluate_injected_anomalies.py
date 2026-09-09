"""Evaluate LSTM detection on synthetic AWS anomalies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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

# Number of normal sequences used for the experiment.
N_TEST_SEQUENCES = 500


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

data = prepare_lstm_data(
    str(CSV_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


# IMPORTANT:
# data.X_eval is already scaled.
# We need raw physical sensor values for anomaly injection.
#
# Reconstruct raw values using the fitted scaler.

n_samples, n_steps, n_features = data.X_eval.shape

raw_eval = data.scaler.inverse_transform(
    data.X_eval.reshape(-1, n_features)
).reshape(
    n_samples,
    n_steps,
    n_features,
)


# Limit experiment size
raw_eval = raw_eval[:N_TEST_SEQUENCES]


# ---------------------------------------------------------
# Create detector
# ---------------------------------------------------------

detector = TemporalAnomalyDetector(
    model_path=str(MODEL_PATH),
    scaler_path=str(SCALER_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


# ---------------------------------------------------------
# Create normal sequences
# ---------------------------------------------------------

normal_sequences = raw_eval.copy()


# ---------------------------------------------------------
# Create anomalous sequences
# ---------------------------------------------------------

temperature_spikes = np.array(
    [
        inject_temperature_spike(
            sequence,
            magnitude=20.0,
        )
        for sequence in raw_eval
    ],
    dtype=np.float32,
)


pressure_spikes = np.array(
    [
        inject_pressure_spike(
            sequence,
            magnitude=40.0,
        )
        for sequence in raw_eval
    ],
    dtype=np.float32,
)


humidity_spikes = np.array(
    [
        inject_humidity_spike(
            sequence,
            value=100.0,
        )
        for sequence in raw_eval
    ],
    dtype=np.float32,
)


frozen_sensors = np.array(
    [
        inject_frozen_temperature(
            sequence,
            duration=8,
        )
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


multivariate_inconsistencies = np.array(
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
# Calculate reconstruction errors
# ---------------------------------------------------------

print("\nCalculating reconstruction errors...\n")


def calculate_errors(
    sequences: np.ndarray,
) -> np.ndarray:

    return detector.calculate_reconstruction_error(
        sequences,
        already_scaled=False,
    )


results = {
    "NORMAL": calculate_errors(normal_sequences),

    "TEMPERATURE_SPIKE": calculate_errors(
        temperature_spikes
    ),

    "PRESSURE_SPIKE": calculate_errors(
        pressure_spikes
    ),

    "HUMIDITY_SPIKE": calculate_errors(
        humidity_spikes
    ),

    "FROZEN_SENSOR": calculate_errors(
        frozen_sensors
    ),

    "CALIBRATION_DRIFT": calculate_errors(
        calibration_drifts
    ),

    "MULTIVARIATE_INCONSISTENCY": calculate_errors(
        multivariate_inconsistencies
    ),
}


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("=" * 70)
print("LSTM ANOMALY INJECTION EVALUATION")
print("=" * 70)

print(
    f"\nSequences per category: "
    f"{N_TEST_SEQUENCES}"
)

print(
    "\n"
    f"{'Anomaly Type':<30}"
    f"{'Mean':>12}"
    f"{'Median':>12}"
    f"{'95%':>12}"
    f"{'Max':>12}"
)

print("-" * 78)


for name, errors in results.items():

    print(
        f"{name:<30}"
        f"{np.mean(errors):>12.6f}"
        f"{np.median(errors):>12.6f}"
        f"{np.percentile(errors, 95):>12.6f}"
        f"{np.max(errors):>12.6f}"
    )


print("=" * 70)
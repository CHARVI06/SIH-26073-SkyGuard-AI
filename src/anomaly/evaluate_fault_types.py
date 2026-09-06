"""Evaluate LSTM detection performance for each injected fault type."""

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

SEQUENCE_LENGTH = 24
N_SEQUENCES = 500

# Candidate threshold from previous experiment
THRESHOLD = 0.08


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

data = prepare_lstm_data(
    str(CSV_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


# ---------------------------------------------------------
# Convert evaluation sequences to physical values
# ---------------------------------------------------------

n_samples, n_steps, n_features = data.X_eval.shape

raw_eval = data.scaler.inverse_transform(
    data.X_eval.reshape(-1, n_features)
).reshape(
    n_samples,
    n_steps,
    n_features,
)

raw_eval = raw_eval[:N_SEQUENCES]


# ---------------------------------------------------------
# Detector
# ---------------------------------------------------------

detector = TemporalAnomalyDetector(
    model_path=str(MODEL_PATH),
    scaler_path=str(SCALER_PATH),
    sequence_length=SEQUENCE_LENGTH,
    threshold=THRESHOLD,
)


# ---------------------------------------------------------
# Define fault injection functions
# ---------------------------------------------------------

faults = {
    "TEMPERATURE_SPIKE": lambda x: inject_temperature_spike(
        x,
        magnitude=20.0,
    ),

    "PRESSURE_SPIKE": lambda x: inject_pressure_spike(
        x,
        magnitude=40.0,
    ),

    "HUMIDITY_SPIKE": lambda x: inject_humidity_spike(
        x,
        value=100.0,
    ),

    "FROZEN_SENSOR": lambda x: inject_frozen_temperature(
        x,
        duration=8,
    ),

    "CALIBRATION_DRIFT": lambda x: inject_calibration_drift(
        x,
        drift_per_step=0.5,
    ),

    "MULTIVARIATE_INCONSISTENCY": lambda x: (
        inject_multivariate_inconsistency(
            x,
            temperature_offset=15.0,
        )
    ),
}


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("PER-FAULT LSTM DETECTION EVALUATION")
print("=" * 85)

print(f"\nThreshold: {THRESHOLD}")
print(f"Sequences per fault: {N_SEQUENCES}\n")

print(
    f"{'Fault Type':<32}"
    f"{'Detected':>12}"
    f"{'Missed':>10}"
    f"{'Detection Rate':>18}"
)

print("-" * 85)


results = {}


for fault_name, inject_function in faults.items():

    injected = np.array(
        [
            inject_function(sequence)
            for sequence in raw_eval
        ],
        dtype=np.float32,
    )

    predictions = detector.predict(
        injected,
        already_scaled=False,
    )

    detected = int(
        np.sum(predictions["is_anomaly"])
    )

    missed = N_SEQUENCES - detected

    detection_rate = (
        detected / N_SEQUENCES
    )

    results[fault_name] = {
        "detected": detected,
        "missed": missed,
        "detection_rate": detection_rate,
    }

    print(
        f"{fault_name:<32}"
        f"{detected:>12}"
        f"{missed:>10}"
        f"{detection_rate * 100:>17.2f}%"
    )


print("-" * 85)


average_detection_rate = np.mean(
    [
        result["detection_rate"]
        for result in results.values()
    ]
)

print(
    f"{'AVERAGE':<32}"
    f"{'':>12}"
    f"{'':>10}"
    f"{average_detection_rate * 100:>17.2f}%"
)

print("=" * 85)
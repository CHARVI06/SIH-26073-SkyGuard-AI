"""Analyze reconstruction-error distribution of the trained LSTM."""

from pathlib import Path

import numpy as np

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


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

data = prepare_lstm_data(
    str(CSV_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


# ---------------------------------------------------------
# Create detector
# ---------------------------------------------------------

detector = TemporalAnomalyDetector(
    model_path=str(MODEL_PATH),
    scaler_path=str(SCALER_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


# ---------------------------------------------------------
# Calculate reconstruction errors
# ---------------------------------------------------------

print("\nCalculating reconstruction errors...\n")

train_errors = detector.calculate_reconstruction_error(
    data.X_train,
    already_scaled=True,
)

eval_errors = detector.calculate_reconstruction_error(
    data.X_eval,
    already_scaled=True,
)


# ---------------------------------------------------------
# Display statistics
# ---------------------------------------------------------

print("Training reconstruction errors")
print("--------------------------------")

print(f"Count  : {len(train_errors)}")
print(f"Minimum: {np.min(train_errors):.6f}")
print(f"Mean   : {np.mean(train_errors):.6f}")
print(f"Median : {np.median(train_errors):.6f}")
print(f"Maximum: {np.max(train_errors):.6f}")

print("\nPercentiles:")

for percentile in [90, 95, 97, 98, 99, 99.5]:
    value = np.percentile(
        train_errors,
        percentile,
    )

    print(
        f"{percentile:>5}% : {value:.6f}"
    )


print("\nEvaluation reconstruction errors")
print("---------------------------------")

print(f"Count  : {len(eval_errors)}")
print(f"Minimum: {np.min(eval_errors):.6f}")
print(f"Mean   : {np.mean(eval_errors):.6f}")
print(f"Median : {np.median(eval_errors):.6f}")
print(f"Maximum: {np.max(eval_errors):.6f}")

print("\nPercentiles:")

for percentile in [90, 95, 97, 98, 99, 99.5]:
    value = np.percentile(
        eval_errors,
        percentile,
    )

    print(
        f"{percentile:>5}% : {value:.6f}"
    )
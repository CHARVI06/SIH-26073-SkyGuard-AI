"""Evaluate Multivariate Consistency v2.3 on controlled anomalies."""

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
from src.anomaly.multivariate_check import (
    MultivariateConsistencyChecker,
)
from src.models.sequence_data import prepare_lstm_data


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imd_maitri_member1_features.csv"
)

SEQUENCE_LENGTH = 24
N_SEQUENCES = 500

# Point anomalies are injected at the middle
# of the 24-hour sequence.
INJECTION_POSITION = SEQUENCE_LENGTH // 2


# =========================================================
# Load sequence data
# =========================================================

print("\nLoading dataset...")

data = prepare_lstm_data(
    str(CSV_PATH),
    sequence_length=SEQUENCE_LENGTH,
)

print(
    f"Training sequences : {len(data.X_train)}"
)

print(
    f"Evaluation sequences: {len(data.X_eval)}"
)


# =========================================================
# Convert scaled sequences back to physical values
# =========================================================

n_train, n_steps, n_features = (
    data.X_train.shape
)

raw_train = data.scaler.inverse_transform(
    data.X_train.reshape(
        -1,
        n_features,
    )
).reshape(
    n_train,
    n_steps,
    n_features,
)


n_eval, n_steps, n_features = (
    data.X_eval.shape
)

raw_eval = data.scaler.inverse_transform(
    data.X_eval.reshape(
        -1,
        n_features,
    )
).reshape(
    n_eval,
    n_steps,
    n_features,
)


# Convert to float32.
raw_train = raw_train.astype(
    np.float32
)

raw_eval = raw_eval.astype(
    np.float32
)


# Use only the requested number of
# evaluation sequences.
raw_eval = raw_eval[
    :N_SEQUENCES
]


# =========================================================
# Fit Multivariate Consistency Checker
# =========================================================

print("\nFitting multivariate consistency checker...")

checker = MultivariateConsistencyChecker(
    change_z_threshold=3.0,
    isolation_ratio=2.0,
    score_threshold=0.70,
)

checker.fit(
    raw_train
)


# =========================================================
# Evaluation helper
# =========================================================

def evaluate_sequences(
    sequences: np.ndarray,
) -> tuple[int, float, list[float]]:
    """Evaluate complete 24-hour sequences."""

    detected = 0

    scores: list[float] = []

    for sequence in sequences:

        result = checker.check(
            sequence
        )

        scores.append(
            result.score
        )

        if result.is_inconsistent:
            detected += 1

    if len(sequences) == 0:
        return 0, 0.0, []

    detection_rate = (
        detected
        / len(sequences)
    )

    return (
        detected,
        detection_rate,
        scores,
    )


# =========================================================
# Fault definitions
# =========================================================

faults = {

    "TEMPERATURE_SPIKE": lambda x:
        inject_temperature_spike(
            x,
            magnitude=20.0,
            position=INJECTION_POSITION,
        ),

    "PRESSURE_SPIKE": lambda x:
        inject_pressure_spike(
            x,
            magnitude=40.0,
            position=INJECTION_POSITION,
        ),

    "HUMIDITY_SPIKE": lambda x:
        inject_humidity_spike(
            x,
            value=100.0,
            position=INJECTION_POSITION,
        ),

    "FROZEN_SENSOR": lambda x:
        inject_frozen_temperature(
            x,
            duration=8,
            position=INJECTION_POSITION,
        ),

    "CALIBRATION_DRIFT": lambda x:
        inject_calibration_drift(
            x,
            drift_per_step=0.5,
        ),

    "MULTIVARIATE_INCONSISTENCY": lambda x:
        inject_multivariate_inconsistency(
            x,
            temperature_offset=15.0,
            position=INJECTION_POSITION,
        ),
}


# =========================================================
# Evaluation header
# =========================================================

print("\n" + "=" * 100)

print(
    "MULTIVARIATE CONSISTENCY V2.3 EVALUATION"
)

print("=" * 100)

print(
    f"\nSequence length       : "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Injection position    : "
    f"{INJECTION_POSITION}"
)

print(
    f"Sequences per fault   : "
    f"{N_SEQUENCES}"
)

print(
    f"Change Z threshold    : "
    f"{checker.change_z_threshold}"
)

print(
    f"Isolation ratio       : "
    f"{checker.isolation_ratio}"
)

print(
    f"Score threshold       : "
    f"{checker.score_threshold}"
)

print()


# =========================================================
# Table header
# =========================================================

print(
    f"{'Fault Type':<35}"
    f"{'Detected':>12}"
    f"{'Missed':>10}"
    f"{'Detection Rate':>18}"
    f"{'Mean Score':>15}"
)

print("-" * 100)


# =========================================================
# Normal baseline
# =========================================================

normal_detected, normal_rate, normal_scores = (
    evaluate_sequences(
        raw_eval
    )
)

normal_missed = (
    N_SEQUENCES
    - normal_detected
)

normal_mean_score = (
    np.mean(normal_scores)
    if normal_scores
    else 0.0
)

print(
    f"{'NORMAL (False Positive)':<35}"
    f"{normal_detected:>12}"
    f"{normal_missed:>10}"
    f"{normal_rate * 100:>17.2f}%"
    f"{normal_mean_score:>15.4f}"
)


# =========================================================
# Fault evaluation
# =========================================================

for (
    fault_name,
    inject_function,
) in faults.items():

    # Inject the selected fault into every
    # evaluation sequence.
    injected = np.array(
        [
            inject_function(
                sequence
            )
            for sequence in raw_eval
        ],
        dtype=np.float32,
    )

    detected, detection_rate, scores = (
        evaluate_sequences(
            injected
        )
    )

    missed = (
        N_SEQUENCES
        - detected
    )

    mean_score = (
        np.mean(scores)
        if scores
        else 0.0
    )

    print(
        f"{fault_name:<35}"
        f"{detected:>12}"
        f"{missed:>10}"
        f"{detection_rate * 100:>17.2f}%"
        f"{mean_score:>15.4f}"
    )


# =========================================================
# Finish
# =========================================================

print("=" * 100)

print(
    "\nEvaluation completed successfully."
)
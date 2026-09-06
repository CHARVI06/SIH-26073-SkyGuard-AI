"""Calibrate multivariate anomaly threshold using normal AWS data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
N_EVAL_SEQUENCES = 500

# Candidate thresholds to investigate.
CANDIDATE_THRESHOLDS = [
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
]


# =========================================================
# Load data
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
# Convert scaled data back to physical values
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
).astype(np.float32)


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
).astype(np.float32)


# =========================================================
# Limit evaluation set
# =========================================================

raw_eval = raw_eval[
    :N_EVAL_SEQUENCES
]


# =========================================================
# Fit detector
# =========================================================

print(
    "\nFitting multivariate consistency checker..."
)

checker = MultivariateConsistencyChecker(
    change_z_threshold=4.0,
    isolation_ratio=3.0,
    score_threshold=4.0,
)

checker.fit(
    raw_train
)


# =========================================================
# Collect training scores
# =========================================================

print(
    "\nCalculating normal training scores..."
)

training_scores = []

for sequence in raw_train:

    result = checker.check(
        sequence
    )

    training_scores.append(
        result.score
    )

training_scores = np.asarray(
    training_scores,
    dtype=np.float64,
)


# =========================================================
# Score statistics
# =========================================================

percentiles = [
    90,
    95,
    97,
    98,
    99,
    99.5,
]

print("\n" + "=" * 70)

print(
    "NORMAL TRAINING SCORE DISTRIBUTION"
)

print("=" * 70)

print(
    f"Count   : {len(training_scores)}"
)

print(
    f"Minimum : {np.min(training_scores):.6f}"
)

print(
    f"Mean    : {np.mean(training_scores):.6f}"
)

print(
    f"Median  : {np.median(training_scores):.6f}"
)

for percentile in percentiles:

    value = np.percentile(
        training_scores,
        percentile,
    )

    print(
        f"P{percentile:<5}: "
        f"{value:.6f}"
    )

print(
    f"Maximum : {np.max(training_scores):.6f}"
)


# =========================================================
# Evaluate thresholds on normal evaluation data
# =========================================================

print("\n" + "=" * 70)

print(
    "THRESHOLD TEST ON NORMAL EVALUATION DATA"
)

print("=" * 70)

print()

print(
    f"{'Threshold':<15}"
    f"{'False Positives':<20}"
    f"{'False Positive Rate':<20}"
)

print("-" * 55)


evaluation_scores = []

for sequence in raw_eval:

    result = checker.check(
        sequence
    )

    evaluation_scores.append(
        result.score
    )

evaluation_scores = np.asarray(
    evaluation_scores,
    dtype=np.float64,
)


for threshold in CANDIDATE_THRESHOLDS:

    false_positives = int(
        np.sum(
            evaluation_scores
            >= threshold
        )
    )

    false_positive_rate = (
        false_positives
        / len(evaluation_scores)
    )

    print(
        f"{threshold:<15.2f}"
        f"{false_positives:<20}"
        f"{false_positive_rate * 100:>17.2f}%"
    )


# =========================================================
# Finish
# =========================================================

print("=" * 70)

print(
    "\nCalibration completed successfully."
)
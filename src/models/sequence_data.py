"""Prepare AWS sensor sequences for the LSTM Autoencoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SENSOR_COLUMNS = (
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",
)


@dataclass
class SequenceData:
    """Container for prepared LSTM training and evaluation data."""

    X_train: np.ndarray
    X_eval: np.ndarray
    scaler: StandardScaler


def create_sequences(
    frame: pd.DataFrame,
    sequence_length: int = 24,
) -> np.ndarray:
    """Create fixed-length sequences without crossing segment boundaries."""

    sequences = []

    for _, segment in frame.groupby("segment_id", sort=False):
        values = segment.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float32)

        if len(values) < sequence_length:
            continue

        for start in range(len(values) - sequence_length + 1):
            end = start + sequence_length
            sequences.append(values[start:end])

    if not sequences:
        raise ValueError("No valid sequences could be created.")

    return np.asarray(sequences, dtype=np.float32)


def prepare_lstm_data(
    csv_path: str,
    sequence_length: int = 24,
    train_ratio: float = 0.8,
) -> SequenceData:
    """Load the processed dataset and prepare chronological LSTM data."""

    frame = pd.read_csv(csv_path)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    # Create sequences without crossing segment boundaries.
    sequences = create_sequences(
        frame,
        sequence_length=sequence_length,
    )

    # Chronological split.
    split_index = int(len(sequences) * train_ratio)

    X_train = sequences[:split_index]
    X_eval = sequences[split_index:]

    if len(X_train) == 0 or len(X_eval) == 0:
        raise ValueError("Training or evaluation set is empty.")

    # Fit scaler ONLY on training data.
    scaler = StandardScaler()

    n_samples, n_steps, n_features = X_train.shape

    X_train_2d = X_train.reshape(-1, n_features)

    scaler.fit(X_train_2d)

    # Apply the training scaler to both sets.
    X_train = scaler.transform(
        X_train.reshape(-1, n_features)
    ).reshape(n_samples, n_steps, n_features)

    eval_samples = X_eval.shape[0]

    X_eval = scaler.transform(
        X_eval.reshape(-1, n_features)
    ).reshape(eval_samples, n_steps, n_features)

    return SequenceData(
        X_train=X_train.astype(np.float32),
        X_eval=X_eval.astype(np.float32),
        scaler=scaler,
    )
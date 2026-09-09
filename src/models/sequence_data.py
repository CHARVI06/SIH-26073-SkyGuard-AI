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
    """Load processed AWS data and prepare leakage-free chronological LSTM data."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")

    frame = pd.read_csv(csv_path)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(
        ["segment_id", "timestamp"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Split the RAW time-series data before creating sliding windows.
    #
    # A gap of sequence_length - 1 observations prevents the last
    # training window from overlapping with the first evaluation window.
    # ------------------------------------------------------------------
    gap = sequence_length - 1

    train_parts = []
    eval_parts = []

    for _, segment in frame.groupby("segment_id", sort=False):
        segment = segment.sort_values("timestamp").reset_index(drop=True)

        split_index = int(len(segment) * train_ratio)

        train_segment = segment.iloc[:split_index]

        eval_start = split_index + gap
        eval_segment = segment.iloc[eval_start:]

        if len(train_segment) >= sequence_length:
            train_parts.append(train_segment)

        if len(eval_segment) >= sequence_length:
            eval_parts.append(eval_segment)

    if not train_parts or not eval_parts:
        raise ValueError(
            "Not enough data to create training and evaluation sequences "
            "after applying the temporal gap."
        )

    train_frame = pd.concat(train_parts, ignore_index=True)
    eval_frame = pd.concat(eval_parts, ignore_index=True)

    # ------------------------------------------------------------------
    # Create sequences AFTER the chronological split.
    # ------------------------------------------------------------------
    X_train = create_sequences(
        train_frame,
        sequence_length=sequence_length,
    )

    X_eval = create_sequences(
        eval_frame,
        sequence_length=sequence_length,
    )

    if len(X_train) == 0 or len(X_eval) == 0:
        raise ValueError("Training or evaluation set is empty.")

    # ------------------------------------------------------------------
    # Fit scaler ONLY on training data.
    # ------------------------------------------------------------------
    scaler = StandardScaler()

    n_train, n_steps, n_features = X_train.shape

    X_train_2d = X_train.reshape(-1, n_features)

    scaler.fit(X_train_2d)

    # Transform training data using training-only statistics.
    X_train = scaler.transform(
        X_train_2d
    ).reshape(n_train, n_steps, n_features)

    # Transform evaluation data using the SAME training scaler.
    n_eval = X_eval.shape[0]

    X_eval = scaler.transform(
        X_eval.reshape(-1, n_features)
    ).reshape(n_eval, n_steps, n_features)

    return SequenceData(
        X_train=X_train.astype(np.float32),
        X_eval=X_eval.astype(np.float32),
        scaler=scaler,
    )
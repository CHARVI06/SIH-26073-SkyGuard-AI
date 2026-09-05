"""Validation and quality-summary helpers for raw AWS observations."""
from __future__ import annotations

import pandas as pd

from .load_data import REQUIRED_SOURCE_COLUMNS

MISSING_SENTINEL = -999


def validate_source_frame(frame: pd.DataFrame) -> None:
    """Raise a clear error when a source frame cannot support the baseline."""
    absent = set(REQUIRED_SOURCE_COLUMNS).difference(frame.columns)
    if absent:
        raise ValueError(f"Dataset is missing required columns: {sorted(absent)}")
    if frame.empty:
        raise ValueError("Dataset contains no observations.")
    if frame["timestamp"].isna().any():
        raise ValueError("Dataset contains unparsable timestamps.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Dataset timestamps must be sorted before validation.")


def quality_report(frame: pd.DataFrame) -> dict[str, object]:
    """Return serialisable source-data facts, preserving sentinel semantics."""
    validate_source_frame(frame)
    target_columns = ["tempr", "rh", "ap"]
    intervals = frame["timestamp"].diff().dt.total_seconds().div(60).dropna()
    valid = frame[target_columns].replace(MISSING_SENTINEL, pd.NA)
    return {
        "rows": int(len(frame)),
        "time_start": frame["timestamp"].min().isoformat(),
        "time_end": frame["timestamp"].max().isoformat(),
        "duplicate_timestamps": int(frame["timestamp"].duplicated().sum()),
        "missing_timestamps": int(frame["timestamp"].isna().sum()),
        "missing_sentinel_counts": {column: int(frame[column].eq(MISSING_SENTINEL).sum()) for column in target_columns},
        "sampling_intervals_minutes": {str(key): int(value) for key, value in intervals.value_counts().sort_index().items()},
        "valid_ranges": {
            column: {"min": float(valid[column].min()), "max": float(valid[column].max())}
            for column in target_columns
        },
    }

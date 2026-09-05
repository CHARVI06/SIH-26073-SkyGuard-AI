"""Leakage-safe trailing rolling statistics for the baseline model."""
from __future__ import annotations

import pandas as pd


def add_rolling_features(frame: pd.DataFrame, columns: tuple[str, ...], windows: tuple[int, ...] = (6, 24)) -> pd.DataFrame:
    """Add within-segment trailing means, standard deviations, and deltas.

    The current value is not used in its own rolling reference window, avoiding
    leakage when the model scores a live observation.
    """
    if "segment_id" not in frame:
        raise ValueError("Rolling features require segment_id from add_temporal_features.")
    absent = set(columns).difference(frame.columns)
    if absent:
        raise ValueError(f"Cannot build rolling features; missing: {sorted(absent)}")
    out = frame.copy()
    for column in columns:
        grouped = out.groupby("segment_id", sort=False)[column]
        out[f"{column}_delta_1"] = grouped.diff()
        for window in windows:
            prior = grouped.transform(lambda series: series.shift(1).rolling(window, min_periods=window).mean())
            prior_std = grouped.transform(lambda series: series.shift(1).rolling(window, min_periods=window).std())
            out[f"{column}_rolling_mean_{window}"] = prior
            out[f"{column}_rolling_std_{window}"] = prior_std.fillna(0.0)
    return out

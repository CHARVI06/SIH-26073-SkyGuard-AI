"""Calendar and continuity features derived from observation timestamps."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(frame: pd.DataFrame, max_gap_hours: float = 3.0) -> pd.DataFrame:
    """Add cyclical calendar fields and ``segment_id`` reset after data outages."""
    if "timestamp" not in frame:
        raise ValueError("Temporal features require a timestamp column.")
    out = frame.sort_values("timestamp").copy()
    timestamp = pd.to_datetime(out["timestamp"], errors="raise")
    out["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    out["dayofyear_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofyear / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofyear / 365.25)
    gap_hours = timestamp.diff().dt.total_seconds().div(3600).fillna(0)
    out["segment_id"] = (gap_hours > max_gap_hours).cumsum().astype(int)
    return out

"""Stable Member 1 feature contract shared with downstream team members."""
from __future__ import annotations

from src.data.preprocess import BASE_VARIABLES

TEMPORAL_FEATURES = (
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
)


def rolling_feature_names(
    variables: tuple[str, ...] = BASE_VARIABLES,
    windows: tuple[int, ...] = (6, 24),
) -> tuple[str, ...]:
    """Return the fixed causal rolling-feature names in pipeline order."""
    names: list[str] = []
    for variable in variables:
        names.append(f"{variable}_delta_1")
        for window in windows:
            names.extend((f"{variable}_rolling_mean_{window}", f"{variable}_rolling_std_{window}"))
    return tuple(names)


MODEL_FEATURE_COLUMNS = BASE_VARIABLES + TEMPORAL_FEATURES + rolling_feature_names()


def select_model_features(frame_columns) -> list[str]:
    """Validate and return the immutable order used by the Isolation Forest."""
    absent = set(MODEL_FEATURE_COLUMNS).difference(frame_columns)
    if absent:
        raise ValueError(f"Feature frame is missing model features: {sorted(absent)}")
    return list(MODEL_FEATURE_COLUMNS)

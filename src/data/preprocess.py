"""Reproducible cleaning for the three verified baseline weather variables."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .load_data import load_imd_csv
from .validate import MISSING_SENTINEL, validate_source_frame

BASE_VARIABLES = ("temperature_c", "pressure_hpa", "humidity_pct")
SOURCE_TO_BASE = {"tempr": "temperature_c", "rh": "pressure_hpa", "ap": "humidity_pct"}


@dataclass(frozen=True)
class PreprocessConfig:
    max_interpolation_steps: int = 3


def preprocess_observations(raw: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """Clean source observations without filling across long outages.

    ``-999`` is changed to NaN, short internal runs are time-interpolated, and
    remaining incomplete rows are excluded from model-ready output.  A row-level
    ``was_imputed`` flag retains this information for downstream consumers.
    """
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="first").copy()
    validate_source_frame(raw)
    clean = raw.loc[:, ["timestamp", *SOURCE_TO_BASE]].rename(columns=SOURCE_TO_BASE)
    # CSV wind/humidity-derived columns can be integer typed; promote before
    # replacing the integer sentinel with NaN (required by pandas 3+).
    clean = clean.astype({column: float for column in BASE_VARIABLES})
    clean.loc[:, BASE_VARIABLES] = clean.loc[:, BASE_VARIABLES].replace(MISSING_SENTINEL, float("nan"))
    clean["was_imputed"] = clean.loc[:, BASE_VARIABLES].isna().any(axis=1)
    clean = clean.set_index("timestamp")
    # limit_area prevents invented leading/trailing readings; no reindexing means
    # gaps remain gaps rather than becoming fabricated hourly observations.
    clean.loc[:, BASE_VARIABLES] = clean.loc[:, BASE_VARIABLES].interpolate(
        method="time", limit=config.max_interpolation_steps, limit_area="inside"
    )
    clean = clean.dropna(subset=BASE_VARIABLES).reset_index()
    if clean.empty:
        raise ValueError("No complete observations remain after preprocessing.")
    return clean


def preprocess_csv(path: str | None = None, config: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    return preprocess_observations(load_imd_csv(path), config)

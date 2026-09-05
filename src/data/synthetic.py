"""Deterministic synthetic anomaly scenarios for supervised baseline evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ("temperature_c", "pressure_hpa", "humidity_pct")


def inject_synthetic_anomalies(frame: pd.DataFrame, random_state: int = 42, per_type: int = 20) -> pd.DataFrame:
    """Inject labelled spikes, drops, freezes, drifts, and missing readings.

    Injection is deterministic and avoids edges so rolling features remain valid.
    The original clean frame is never mutated.
    """
    if len(frame) < 200:
        raise ValueError("At least 200 clean observations are needed for synthetic scenarios.")
    out = frame.copy()
    out["is_synthetic_anomaly"] = False
    out["anomaly_type"] = "normal"
    out["affected_feature"] = ""
    rng = np.random.default_rng(random_state)
    scales = out.loc[:, TARGETS].std().replace(0, 1.0)
    scenarios = ("spike", "drop", "freeze", "drift", "missing")
    candidates = np.arange(48, len(out) - 48)
    starts = rng.choice(candidates, size=per_type * len(scenarios), replace=False)
    for index, (scenario, start) in enumerate(zip(np.repeat(scenarios, per_type), starts)):
        column = TARGETS[index % len(TARGETS)]
        length = 1 if scenario in {"spike", "drop"} else 6
        rows = out.index[start : start + length]
        if scenario == "spike":
            out.loc[rows, column] += 6 * scales[column]
        elif scenario == "drop":
            out.loc[rows, column] -= 6 * scales[column]
        elif scenario == "freeze":
            out.loc[rows, column] = out.loc[rows[0], column]
        elif scenario == "drift":
            out.loc[rows, column] += np.linspace(1, 4, length) * scales[column]
        else:
            out.loc[rows, column] = np.nan
        out.loc[rows, ["is_synthetic_anomaly", "anomaly_type", "affected_feature"]] = [True, scenario, column]
    return out

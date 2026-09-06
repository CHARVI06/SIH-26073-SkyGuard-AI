"""Inject controlled synthetic AWS sensor anomalies for evaluation."""

from __future__ import annotations

import numpy as np


SENSOR_INDEX = {
    "temperature": 0,
    "pressure": 1,
    "humidity": 2,
}


def inject_temperature_spike(
    sequence: np.ndarray,
    magnitude: float = 20.0,
    position: int | None = None,
) -> np.ndarray:
    """Inject a sudden temperature spike."""

    modified = sequence.copy()

    if position is None:
        position = len(modified) // 2

    modified[position, SENSOR_INDEX["temperature"]] += magnitude

    return modified


def inject_pressure_spike(
    sequence: np.ndarray,
    magnitude: float = 40.0,
    position: int | None = None,
) -> np.ndarray:
    """Inject a sudden atmospheric-pressure spike."""

    modified = sequence.copy()

    if position is None:
        position = len(modified) // 2

    modified[position, SENSOR_INDEX["pressure"]] += magnitude

    return modified


def inject_humidity_spike(
    sequence: np.ndarray,
    value: float = 100.0,
    position: int | None = None,
) -> np.ndarray:
    """Inject an abnormal humidity reading."""

    modified = sequence.copy()

    if position is None:
        position = len(modified) // 2

    modified[position, SENSOR_INDEX["humidity"]] = value

    return modified


def inject_frozen_temperature(
    sequence: np.ndarray,
    duration: int = 8,
    position: int | None = None,
) -> np.ndarray:
    """Simulate a temperature sensor becoming frozen."""

    modified = sequence.copy()

    if position is None:
        position = len(modified) // 2

    start = max(0, position - duration // 2)
    end = min(len(modified), start + duration)

    frozen_value = modified[start, SENSOR_INDEX["temperature"]]

    modified[
        start:end,
        SENSOR_INDEX["temperature"],
    ] = frozen_value

    return modified


def inject_calibration_drift(
    sequence: np.ndarray,
    drift_per_step: float = 0.5,
) -> np.ndarray:
    """Simulate gradual temperature calibration drift."""

    modified = sequence.copy()

    temperature_index = SENSOR_INDEX["temperature"]

    for i in range(len(modified)):
        modified[i, temperature_index] += (
            i * drift_per_step
        )

    return modified


def inject_multivariate_inconsistency(
    sequence: np.ndarray,
    temperature_offset: float = 15.0,
    position: int | None = None,
) -> np.ndarray:
    """Create an inconsistency between temperature and humidity."""

    modified = sequence.copy()

    if position is None:
        position = len(modified) // 2

    modified[
        position,
        SENSOR_INDEX["temperature"],
    ] += temperature_offset

    return modified
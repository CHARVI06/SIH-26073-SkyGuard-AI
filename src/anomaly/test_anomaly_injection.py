"""Test synthetic AWS anomaly injection."""

import numpy as np

from src.anomaly.inject_anomalies import (
    inject_calibration_drift,
    inject_frozen_temperature,
    inject_humidity_spike,
    inject_multivariate_inconsistency,
    inject_pressure_spike,
    inject_temperature_spike,
)


# ---------------------------------------------------------
# Create a realistic 24-hour example
# ---------------------------------------------------------

sequence = np.array(
    [
        [20.0, 968.0, 60.0],
        [19.8, 967.8, 61.0],
        [19.5, 967.5, 62.0],
        [19.2, 967.2, 63.0],
        [19.0, 967.0, 64.0],
        [18.8, 966.8, 65.0],
        [18.7, 966.7, 66.0],
        [18.6, 966.6, 67.0],
        [18.5, 966.5, 68.0],
        [18.6, 966.6, 67.0],
        [18.8, 966.8, 66.0],
        [19.0, 967.0, 65.0],
        [19.5, 967.5, 64.0],
        [20.0, 968.0, 63.0],
        [20.5, 968.5, 62.0],
        [21.0, 969.0, 61.0],
        [21.5, 969.5, 60.0],
        [22.0, 970.0, 59.0],
        [22.5, 970.5, 58.0],
        [23.0, 971.0, 57.0],
        [22.5, 970.5, 58.0],
        [22.0, 970.0, 59.0],
        [21.5, 969.5, 60.0],
        [21.0, 969.0, 61.0],
    ],
    dtype=np.float32,
)


print("Original shape:", sequence.shape)


# ---------------------------------------------------------
# Test each anomaly
# ---------------------------------------------------------

temperature_spike = inject_temperature_spike(
    sequence,
    magnitude=20.0,
)

pressure_spike = inject_pressure_spike(
    sequence,
    magnitude=40.0,
)

humidity_spike = inject_humidity_spike(
    sequence,
    value=100.0,
)

frozen_temperature = inject_frozen_temperature(
    sequence,
    duration=8,
)

calibration_drift = inject_calibration_drift(
    sequence,
    drift_per_step=0.5,
)

multivariate = inject_multivariate_inconsistency(
    sequence,
    temperature_offset=15.0,
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\nTemperature spike:")
print(temperature_spike[12])

print("\nPressure spike:")
print(pressure_spike[12])

print("\nHumidity spike:")
print(humidity_spike[12])

print("\nFrozen temperature:")
print(frozen_temperature[8:16, 0])

print("\nCalibration drift:")
print(calibration_drift[:, 0])

print("\nMultivariate inconsistency:")
print(multivariate[12])

print("\nAnomaly injection test completed successfully.")
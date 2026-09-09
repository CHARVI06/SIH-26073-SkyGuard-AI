"""Continuous multivariate consistency detector for AWS data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultivariateResult:
    """Result produced by the multivariate consistency detector."""

    is_inconsistent: bool
    score: float
    reasons: list[str]
    sensor: str | None = None


class MultivariateConsistencyChecker:
    """Detect localized cross-sensor inconsistencies.

    The detector focuses on whether one sensor behaves
    abnormally relative to the other sensors.

    It is deliberately not responsible for:

        - frozen sensors
        - communication failures
        - long-term calibration drift
        - impossible physical values

    Those conditions belong to dedicated detectors.
    """

    SENSOR_NAMES = (
        "TEMPERATURE",
        "PRESSURE",
        "HUMIDITY",
    )

    def __init__(
        self,
        change_z_threshold: float = 3.0,
        isolation_ratio: float = 2.0,
        score_threshold: float = 0.70,
    ) -> None:

        self.change_z_threshold = (
            change_z_threshold
        )

        self.isolation_ratio = (
            isolation_ratio
        )

        self.score_threshold = (
            score_threshold
        )

        self.change_scales = np.ones(
            3,
            dtype=np.float64,
        )

        self.is_fitted = False

    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        windows: np.ndarray,
    ) -> "MultivariateConsistencyChecker":
        """Learn robust normal change scales."""

        if windows.ndim != 3:
            raise ValueError(
                "Expected windows with shape "
                "(samples, sequence_length, 3)."
            )

        if windows.shape[2] != 3:
            raise ValueError(
                "Expected exactly three sensor features."
            )

        differences = np.diff(
            windows,
            axis=1,
        )

        for sensor_index in range(3):

            values = differences[
                :,
                :,
                sensor_index,
            ].reshape(-1)

            median = np.median(
                values
            )

            mad = np.median(
                np.abs(
                    values - median
                )
            )

            robust_scale = (
                1.4826 * mad
            )

            if robust_scale < 1e-6:

                robust_scale = float(
                    np.std(values)
                )

            self.change_scales[
                sensor_index
            ] = max(
                robust_scale,
                1e-6,
            )

        self.is_fitted = True

        return self

    # =====================================================
    # CHANGE Z-SCORES
    # =====================================================

    def _change_z_scores(
        self,
        window: np.ndarray,
    ) -> np.ndarray:
        """Calculate robust z-scores of sensor changes."""

        differences = np.diff(
            window,
            axis=0,
        )

        return (
            np.abs(differences)
            / self.change_scales
        )

    # =====================================================
    # CONTINUOUS EVENT SCORE
    # =====================================================

    def _calculate_event_score(
        self,
        strongest: float,
        average_other: float,
    ) -> float:
        """Convert isolated sensor deviation into [0, 1]."""

        if strongest < self.change_z_threshold:
            return 0.0

        # How far the strongest sensor is beyond
        # the normal-change threshold.
        excess = (
            strongest
            - self.change_z_threshold
        )

        magnitude_score = (
            excess
            / (
                excess
                + self.change_z_threshold
            )
        )

        # Relative isolation from the other sensors.
        isolation = (
            strongest
            / max(
                average_other,
                1e-6,
            )
        )

        isolation_score = (
            isolation
            / (
                isolation
                + self.isolation_ratio
            )
        )

        # Combine magnitude and isolation.
        score = (
            0.6 * magnitude_score
            + 0.4 * isolation_score
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # =====================================================
    # MAIN CHECK
    # =====================================================

    def check(
        self,
        window: np.ndarray,
    ) -> MultivariateResult:
        """Evaluate one complete multivariate window."""

        if not self.is_fitted:
            raise RuntimeError(
                "Checker has not been fitted."
            )

        if window.ndim != 2:
            raise ValueError(
                "Expected window shape "
                "(sequence_length, 3)."
            )

        if window.shape[1] != 3:
            raise ValueError(
                "Expected three sensor features."
            )

        if window.shape[0] < 3:
            raise ValueError(
                "At least three observations are "
                "required."
            )

        z_scores = (
            self._change_z_scores(
                window
            )
        )

        best_score = 0.0
        best_sensor = None
        best_timestep = None

        # Examine every local change.
        for timestep in range(
            len(z_scores)
        ):

            row = z_scores[
                timestep
            ]

            strongest_index = int(
                np.argmax(row)
            )

            strongest = float(
                row[
                    strongest_index
                ]
            )

            other_values = [
                float(
                    row[index]
                )
                for index in range(3)
                if index != strongest_index
            ]

            average_other = float(
                np.mean(other_values)
            )

            # If multiple sensors move strongly
            # together, this is more likely to be
            # an environmental event.
            strong_sensor_count = int(
                np.sum(
                    row
                    >= self.change_z_threshold
                )
            )

            if strong_sensor_count >= 2:
                continue

            score = (
                self._calculate_event_score(
                    strongest,
                    average_other,
                )
            )

            if score > best_score:

                best_score = score

                best_sensor = (
                    self.SENSOR_NAMES[
                        strongest_index
                    ]
                )

                best_timestep = timestep

        # No suspicious isolated event.
        if best_sensor is None:

            return MultivariateResult(
                is_inconsistent=False,
                score=0.0,
                reasons=[],
                sensor=None,
            )

        reason = (
            f"{best_sensor}"
            "_CROSS_SENSOR_INCONSISTENCY"
        )

        return MultivariateResult(
            is_inconsistent=(
                best_score
                >= self.score_threshold
            ),
            score=best_score,
            reasons=[reason],
            sensor=best_sensor,
        )
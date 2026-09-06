"""Temporal anomaly detection using the trained LSTM Autoencoder."""

from __future__ import annotations

import joblib
import numpy as np
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder


class TemporalAnomalyDetector:
    """Detect temporal anomalies using LSTM reconstruction error."""

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        sequence_length: int = 24,
        threshold: float | None = None,
    ) -> None:

        self.sequence_length = sequence_length

        # Select CPU/GPU automatically
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load trained model
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model = LSTMAutoencoder(
            n_features=checkpoint["n_features"],
            hidden_size=checkpoint["hidden_size"],
            latent_size=checkpoint["latent_size"],
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # Load scaler used during training
        self.scaler = joblib.load(scaler_path)

        # Threshold will be determined later
        self.threshold = threshold

    def calculate_reconstruction_error(
        self,
        sequences: np.ndarray,
        already_scaled: bool = False,
    ) -> np.ndarray:
        """Calculate reconstruction error for each sequence."""

        if sequences.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(samples, sequence_length, features)."
            )

        if sequences.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length "
                f"{self.sequence_length}, "
                f"got {sequences.shape[1]}."
            )

        n_samples, n_steps, n_features = sequences.shape

        # Data from prepare_lstm_data() is already scaled.
        if already_scaled:
            scaled = sequences

        else:
            # Raw sensor values need to be scaled.
            scaled = self.scaler.transform(
                sequences.reshape(-1, n_features)
            ).reshape(
                n_samples,
                n_steps,
                n_features,
            )

        x = torch.tensor(
            scaled,
            dtype=torch.float32,
        ).to(self.device)

        # Inference does not require gradients.
        with torch.no_grad():
            reconstruction = self.model(x)

        reconstruction = reconstruction.cpu().numpy()

        # Mean squared reconstruction error
        errors = np.mean(
            (scaled - reconstruction) ** 2,
            axis=(1, 2),
        )

        return errors

    def predict(
        self,
        sequences: np.ndarray,
        already_scaled: bool = False,
    ) -> dict:
        """Return anomaly predictions and reconstruction errors."""

        errors = self.calculate_reconstruction_error(
            sequences,
            already_scaled=already_scaled,
        )

        if self.threshold is None:
            raise ValueError(
                "Anomaly threshold has not been set."
            )

        predictions = errors > self.threshold

        return {
            "anomaly_scores": errors,
            "is_anomaly": predictions,
        }
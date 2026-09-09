"""LSTM Autoencoder for temporal anomaly detection in AWS sensor data."""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """LSTM-based sequence autoencoder.

    Expected input shape:
        (batch_size, sequence_length, n_features)

    For SkyGuard:
        sequence_length = 24
        n_features = 3
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Compress the encoder representation
        self.to_latent = nn.Linear(hidden_size, latent_size)

        # Convert latent representation back to decoder dimension
        self.from_latent = nn.Linear(latent_size, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Reconstruct the original 3 sensor variables
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input sequence."""

        # x shape:
        # (batch_size, sequence_length, n_features)

        encoded_sequence, _ = self.encoder(x)

        # Take the final time-step representation
        final_hidden = encoded_sequence[:, -1, :]

        # Compress
        latent = self.to_latent(final_hidden)

        # Expand back to decoder size
        decoder_input = self.from_latent(latent)

        # Repeat the representation for every time step
        sequence_length = x.size(1)

        decoder_input = decoder_input.unsqueeze(1).repeat(
            1, sequence_length, 1
        )

        # Decode
        decoded_sequence, _ = self.decoder(decoder_input)

        # Reconstruct original features
        reconstruction = self.output_layer(decoded_sequence)

        return reconstruction
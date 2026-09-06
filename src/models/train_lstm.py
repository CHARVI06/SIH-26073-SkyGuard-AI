"""Train the SkyGuard LSTM Autoencoder."""

from __future__ import annotations

from pathlib import Path

import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.sequence_data import prepare_lstm_data


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imd_maitri_member1_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_autoencoder.pt"
SCALER_PATH = MODEL_DIR / "lstm_scaler.joblib"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEQUENCE_LENGTH = 24
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

data = prepare_lstm_data(
    str(CSV_PATH),
    sequence_length=SEQUENCE_LENGTH,
)


X_train = torch.tensor(data.X_train, dtype=torch.float32)
X_eval = torch.tensor(data.X_eval, dtype=torch.float32)


print("Training data :", X_train.shape)
print("Evaluation data:", X_eval.shape)


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

train_dataset = TensorDataset(X_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = LSTMAutoencoder(
    n_features=3,
    hidden_size=64,
    latent_size=32,
).to(device)


# ---------------------------------------------------------
# Loss and optimizer
# ---------------------------------------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for (batch,) in train_loader:

        batch = batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Reconstruct sequence
        reconstruction = model(batch)

        # Reconstruction error
        loss = criterion(
            reconstruction,
            batch,
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    epoch_loss = total_loss / len(train_dataset)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Training Loss: {epoch_loss:.6f}"
    )


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "n_features": 3,
        "hidden_size": 64,
        "latent_size": 32,
        "sequence_length": SEQUENCE_LENGTH,
    },
    MODEL_PATH,
)


# Save scaler
joblib.dump(
    data.scaler,
    SCALER_PATH,
)


print("\nTraining complete.")

print("Model saved to:")
print(MODEL_PATH)

print("\nScaler saved to:")
print(SCALER_PATH)
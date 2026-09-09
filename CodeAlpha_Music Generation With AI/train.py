"""Train the music LSTM and save weights plus vocabulary."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_midi import MIDI_DIR, generate_dataset  # noqa: E402
from src.model import MusicLSTM  # noqa: E402
from src.preprocess import SEQUENCE_LENGTH, VOCAB_PATH, prepare_dataset  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "music_lstm.pth"

EPOCHS = 18
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
DROPOUT = 0.3


def ensure_midi_dataset() -> None:
    midi_files = list(MIDI_DIR.glob("*.mid"))
    if len(midi_files) >= 20:
        print(f"Found {len(midi_files)} MIDI files in {MIDI_DIR}")
        return
    print("MIDI dataset missing or incomplete — generating synthetic files...")
    generate_dataset(MIDI_DIR)


def train() -> None:
    ensure_midi_dataset()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    x, y, token_to_int, _ = prepare_dataset(
        midi_dir=MIDI_DIR,
        sequence_length=SEQUENCE_LENGTH,
        vocab_path=VOCAB_PATH,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = MusicLSTM(
        vocab_size=len(token_to_int),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        epoch_loss = running_loss / max(batches, 1)
        print(f"Epoch {epoch:02d}/{EPOCHS} - loss: {epoch_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(token_to_int),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": 2,
            "dropout": DROPOUT,
            "sequence_length": SEQUENCE_LENGTH,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vocabulary to {VOCAB_PATH}")


if __name__ == "__main__":
    train()

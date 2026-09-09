"""Stacked LSTM for next-token (note/chord) prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """Embedding → 2-layer LSTM + Dropout → Dense logits (softmax at inference)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        """Return next-token logits of shape (batch, vocab_size)."""
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        last_step = self.dropout(output[:, -1, :])
        logits = self.fc(last_step)
        return logits, hidden

    def predict_proba(self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        """Dense prediction with softmax over the vocabulary."""
        logits, hidden = self.forward(x, hidden)
        return torch.softmax(logits, dim=-1), hidden

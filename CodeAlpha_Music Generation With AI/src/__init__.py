"""AI Music Generation package."""

from .model import MusicLSTM
from .preprocess import SEQUENCE_LENGTH, prepare_dataset

__all__ = ["MusicLSTM", "SEQUENCE_LENGTH", "prepare_dataset"]

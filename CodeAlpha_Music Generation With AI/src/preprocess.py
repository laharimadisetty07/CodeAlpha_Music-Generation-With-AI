"""Parse MIDI files, build a note/chord vocabulary, and create sliding-window sequences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from music21 import chord, converter, note

SEQUENCE_LENGTH = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIDI_DIR = PROJECT_ROOT / "data" / "midi_files"
VOCAB_PATH = PROJECT_ROOT / "checkpoints" / "vocab.json"


def extract_notes(midi_path: Path) -> list[str]:
    """Extract note names (e.g. C4) and chord clusters (e.g. 0.4.7) from a MIDI file."""
    parsed = converter.parse(str(midi_path))
    tokens: list[str] = []
    for element in parsed.flatten().notes:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            tokens.append(".".join(str(pc) for pc in element.normalOrder))
    return tokens


def collect_corpus(midi_dir: Path | None = None) -> list[str]:
    directory = midi_dir or MIDI_DIR
    midi_files = sorted(directory.glob("*.mid")) + sorted(directory.glob("*.midi"))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {directory}")

    corpus: list[str] = []
    for path in midi_files:
        tokens = extract_notes(path)
        print(f"Parsed {path.name}: {len(tokens)} events")
        corpus.extend(tokens)
    if not corpus:
        raise ValueError("Corpus is empty after parsing MIDI files.")
    print(f"Total events: {len(corpus)}")
    return corpus


def build_vocabulary(corpus: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique = sorted(set(corpus))
    token_to_int = {token: i for i, token in enumerate(unique)}
    int_to_token = {i: token for token, i in token_to_int.items()}
    return token_to_int, int_to_token


def create_sequences(
    corpus: list[str],
    token_to_int: dict[str, int],
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: X is sequence_length tokens, y is the next token."""
    encoded = [token_to_int[t] for t in corpus]
    if len(encoded) <= sequence_length:
        raise ValueError(
            f"Need more than {sequence_length} tokens; got {len(encoded)}."
        )

    inputs: list[list[int]] = []
    targets: list[int] = []
    for i in range(len(encoded) - sequence_length):
        inputs.append(encoded[i : i + sequence_length])
        targets.append(encoded[i + sequence_length])

    x = np.array(inputs, dtype=np.int64)
    y = np.array(targets, dtype=np.int64)
    print(f"Sequences: {len(x)} (length={sequence_length}), vocab size={len(token_to_int)}")
    return x, y


def save_vocab(token_to_int: dict[str, int], path: Path | None = None) -> Path:
    path = path or VOCAB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"token_to_int": token_to_int, "sequence_length": SEQUENCE_LENGTH}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved vocabulary ({len(token_to_int)} tokens) to {path}")
    return path


def load_vocab(path: Path | None = None) -> tuple[dict[str, int], dict[int, str], int]:
    path = path or VOCAB_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    token_to_int = payload["token_to_int"]
    int_to_token = {int(i): t for t, i in token_to_int.items()}
    sequence_length = int(payload.get("sequence_length", SEQUENCE_LENGTH))
    return token_to_int, int_to_token, sequence_length


def prepare_dataset(
    midi_dir: Path | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    vocab_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[int, str]]:
    corpus = collect_corpus(midi_dir)
    token_to_int, int_to_token = build_vocabulary(corpus)
    x, y = create_sequences(corpus, token_to_int, sequence_length)
    save_vocab(token_to_int, vocab_path)
    return x, y, token_to_int, int_to_token


if __name__ == "__main__":
    prepare_dataset()

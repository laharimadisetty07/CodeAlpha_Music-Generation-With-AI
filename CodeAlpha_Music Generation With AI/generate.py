"""Load a trained LSTM, generate 100 tokens, and export MIDI."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from music21 import chord, duration, instrument, note, stream, tempo

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MusicLSTM  # noqa: E402
from src.preprocess import (  # noqa: E402
    MIDI_DIR,
    SEQUENCE_LENGTH,
    VOCAB_PATH,
    collect_corpus,
    load_vocab,
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "music_lstm.pth"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_MIDI = OUTPUT_DIR / "generated_music.mid"

GENERATE_LENGTH = 100
TEMPERATURE = 0.9


def token_to_element(token: str, qlen: float = 0.5):
    if "." in token or token.isdigit():
        pcs = [int(p) for p in token.split(".")]
        element = chord.Chord(pcs)
    else:
        element = note.Note(token)
    element.duration = duration.Duration(quarterLength=qlen)
    element.volume.velocity = 80
    return element


def sample_token(probs: torch.Tensor, temperature: float) -> int:
    logits = torch.log(probs.clamp(min=1e-8)) / max(temperature, 1e-6)
    scaled = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(scaled, num_samples=1).item())


def pick_seed(corpus: list[str], token_to_int: dict[str, int], sequence_length: int) -> list[int]:
    if len(corpus) <= sequence_length:
        raise ValueError("Corpus is too short to sample a seed sequence.")
    start = random.randint(0, len(corpus) - sequence_length - 1)
    seed_tokens = corpus[start : start + sequence_length]
    return [token_to_int[t] for t in seed_tokens]


def generate_sequence(
    model: MusicLSTM,
    seed: list[int],
    int_to_token: dict[int, str],
    length: int = GENERATE_LENGTH,
    temperature: float = TEMPERATURE,
    device: torch.device | None = None,
) -> list[str]:
    device = device or next(model.parameters()).device
    model.eval()
    pattern = list(seed)
    generated: list[str] = [int_to_token[i] for i in pattern]

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor(pattern, dtype=torch.long, device=device).unsqueeze(0)
            probs, _ = model.predict_proba(x)
            nxt = sample_token(probs[0], temperature)
            generated.append(int_to_token[nxt])
            pattern = pattern[1:] + [nxt]
    return generated[-length:]


def tokens_to_stream(tokens: list[str]) -> stream.Stream:
    s = stream.Stream()
    s.insert(0, instrument.Piano())
    s.insert(0, tempo.MetronomeMark(number=110))
    for token in tokens:
        qlen = 1.0 if "." in token else 0.5
        s.append(token_to_element(token, qlen=qlen))
    return s


def load_model(vocab_size: int, device: torch.device) -> MusicLSTM:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {MODEL_PATH}. Run train.py first."
        )
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = MusicLSTM(
        vocab_size=ckpt.get("vocab_size", vocab_size),
        embedding_dim=ckpt.get("embedding_dim", 128),
        hidden_size=ckpt.get("hidden_size", 256),
        num_layers=ckpt.get("num_layers", 2),
        dropout=ckpt.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_int, int_to_token, sequence_length = load_vocab(VOCAB_PATH)
    sequence_length = sequence_length or SEQUENCE_LENGTH

    corpus = collect_corpus(MIDI_DIR)
    seed = pick_seed(corpus, token_to_int, sequence_length)
    print(f"Seed length={len(seed)}, vocab={len(token_to_int)}, device={device}")

    model = load_model(len(token_to_int), device)
    generated = generate_sequence(
        model,
        seed,
        int_to_token,
        length=GENERATE_LENGTH,
        temperature=TEMPERATURE,
        device=device,
    )
    print(f"Generated {len(generated)} tokens. Sample: {generated[:12]}")

    midi_stream = tokens_to_stream(generated)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    midi_stream.write("midi", fp=str(OUTPUT_MIDI))
    print(f"Wrote {OUTPUT_MIDI}")


if __name__ == "__main__":
    main()

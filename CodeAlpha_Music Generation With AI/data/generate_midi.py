"""Algorithmically generate a diverse synthetic MIDI dataset with music21."""

from __future__ import annotations

import random
from pathlib import Path

from music21 import chord, duration, instrument, key, meter, note, stream, tempo

ROOT = Path(__file__).resolve().parent
MIDI_DIR = ROOT / "midi_files"

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}

CLASSICAL_PROGRESSIONS = [
    [(0, 4, 7), (5, 9, 12), (7, 11, 14), (0, 4, 7)],
    [(0, 4, 7), (9, 12, 16), (5, 9, 12), (7, 11, 14)],
    [(0, 3, 7), (5, 8, 12), (7, 10, 14), (0, 3, 7)],
]
JAZZ_PROGRESSIONS = [
    [(0, 4, 7, 11), (2, 5, 9, 12), (7, 11, 14, 17), (0, 4, 7, 10)],
    [(0, 3, 7, 10), (5, 8, 12, 15), (7, 10, 14, 17), (0, 3, 7, 10)],
    [(0, 4, 7, 10), (5, 8, 12, 15), (0, 4, 7, 10), (7, 10, 14, 17)],
]

RHYTHMS = {
    "even": [1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
    "syncopated": [0.5, 1.0, 0.5, 0.25, 0.25, 0.5, 1.0],
    "waltz": [1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
    "swing": [0.75, 0.25, 0.75, 0.25, 0.5, 0.5],
    "long": [2.0, 1.0, 1.0, 0.5, 0.5],
}

ROOT_PCS = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "Bb": 10,
}


def _pc_to_midi(pc: int, octave: int) -> int:
    return 12 * (octave + 1) + (pc % 12)


def _append_note(part: stream.Part, midi_pitch: int, qlen: float, velocity: int = 80) -> None:
    midi_pitch = max(36, min(84, midi_pitch))
    n = note.Note(midi_pitch)
    n.duration = duration.Duration(quarterLength=qlen)
    n.volume.velocity = velocity
    part.append(n)


def _append_chord(part: stream.Part, midi_pitches: list[int], qlen: float, velocity: int = 70) -> None:
    c = chord.Chord([max(36, min(84, p)) for p in midi_pitches])
    c.duration = duration.Duration(quarterLength=qlen)
    c.volume.velocity = velocity
    part.append(c)


def _scale_pitches(tonic_pc: int, scale_name: str, octaves: tuple[int, int]) -> list[int]:
    intervals = SCALES[scale_name]
    pitches = []
    for octv in range(octaves[0], octaves[1] + 1):
        for iv in intervals:
            pitches.append(_pc_to_midi(tonic_pc + iv, octv))
    return pitches


def _new_score(bpm: int, time_sig: str, k: key.Key, inst) -> tuple[stream.Score, stream.Part]:
    s = stream.Score()
    part = stream.Part()
    part.insert(0, inst)
    part.insert(0, tempo.MetronomeMark(number=bpm))
    part.insert(0, meter.TimeSignature(time_sig))
    part.insert(0, k)
    s.insert(0, part)
    return s, part


def generate_classical_melody(seed: int, tonic: str, scale_name: str, bpm: int) -> stream.Score:
    rng = random.Random(seed)
    tonic_pc = ROOT_PCS[tonic]
    mode = "minor" if scale_name == "minor" else "major"
    k = key.Key(tonic, mode)
    s, part = _new_score(bpm, "4/4", k, instrument.Piano())
    pitches = _scale_pitches(tonic_pc, scale_name, (3, 5))
    rhythm = RHYTHMS["even"] if rng.random() < 0.5 else RHYTHMS["syncopated"]
    idx = rng.randint(0, len(pitches) - 1)
    for i in range(96):
        step = rng.choice([-2, -1, -1, 0, 1, 1, 2])
        idx = max(0, min(len(pitches) - 1, idx + step))
        qlen = rhythm[i % len(rhythm)]
        vel = rng.randint(64, 100)
        _append_note(part, pitches[idx], qlen, vel)
        if i % 8 == 7 and rng.random() < 0.35:
            triad = [
                pitches[idx],
                pitches[min(len(pitches) - 1, idx + 2)],
                pitches[min(len(pitches) - 1, idx + 4)],
            ]
            _append_chord(part, triad, 1.0, 62)
    return s


def generate_arpeggio_etude(seed: int, tonic: str, scale_name: str, bpm: int) -> stream.Score:
    rng = random.Random(seed)
    tonic_pc = ROOT_PCS[tonic]
    mode = "minor" if scale_name == "minor" else "major"
    k = key.Key(tonic, mode)
    s, part = _new_score(bpm, "4/4", k, instrument.Harpsichord())
    prog = rng.choice(CLASSICAL_PROGRESSIONS)
    octave = rng.choice([3, 4])
    for _ in range(8):
        for triad in prog:
            midi_chord = [_pc_to_midi(tonic_pc + p, octave) for p in triad]
            pattern = list(midi_chord) + [midi_chord[1], midi_chord[0] + 12, midi_chord[2]]
            for p in pattern:
                _append_note(part, p, 0.5, rng.randint(70, 95))
            _append_chord(part, midi_chord, 1.0, 68)
    return s


def generate_waltz(seed: int, tonic: str, scale_name: str, bpm: int) -> stream.Score:
    rng = random.Random(seed)
    tonic_pc = ROOT_PCS[tonic]
    mode = "minor" if scale_name == "minor" else "major"
    k = key.Key(tonic, mode)
    s, part = _new_score(bpm, "3/4", k, instrument.Piano())
    pitches = _scale_pitches(tonic_pc, scale_name, (3, 5))
    bass = _scale_pitches(tonic_pc, scale_name, (2, 3))
    idx = rng.randint(4, len(pitches) - 5)
    for bar in range(24):
        bass_note = bass[bar % len(bass)]
        _append_note(part, bass_note, 1.0, 85)
        for _ in range(2):
            idx = max(0, min(len(pitches) - 1, idx + rng.choice([-1, 1, 1, 2, -2])))
            _append_note(part, pitches[idx], 0.5, rng.randint(60, 90))
        if bar % 4 == 3:
            chord_pcs = CLASSICAL_PROGRESSIONS[0][bar % 4]
            _append_chord(part, [_pc_to_midi(tonic_pc + p, 4) for p in chord_pcs], 1.0, 64)
    return s


def generate_jazz_blues(seed: int, tonic: str, bpm: int) -> stream.Score:
    rng = random.Random(seed)
    tonic_pc = ROOT_PCS[tonic]
    k = key.Key(tonic)
    s, part = _new_score(bpm, "4/4", k, instrument.ElectricPiano())
    blues = _scale_pitches(tonic_pc, "blues", (3, 5))
    rhythm = RHYTHMS["swing"]
    idx = rng.randint(0, len(blues) - 1)
    twelve_bar = [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 7]
    for bar, degree in enumerate(twelve_bar * 3):
        root = _pc_to_midi(tonic_pc + degree, 3)
        seventh = [
            root,
            _pc_to_midi(tonic_pc + degree + 4, 3),
            _pc_to_midi(tonic_pc + degree + 7, 3),
            _pc_to_midi(tonic_pc + degree + 10, 3),
        ]
        _append_chord(part, seventh, 1.0, 72)
        for beat in range(4):
            idx = max(0, min(len(blues) - 1, idx + rng.choice([-2, -1, 0, 1, 1, 2])))
            qlen = rhythm[(bar + beat) % len(rhythm)]
            _append_note(part, blues[idx], qlen, rng.randint(68, 105))
    return s


def generate_jazz_ii_v_i(seed: int, tonic: str, bpm: int) -> stream.Score:
    rng = random.Random(seed)
    tonic_pc = ROOT_PCS[tonic]
    k = key.Key(tonic)
    s, part = _new_score(bpm, "4/4", k, instrument.Saxophone())
    pent = _scale_pitches(tonic_pc, "pentatonic", (3, 5))
    rhythm = RHYTHMS["swing"] if rng.random() < 0.6 else RHYTHMS["syncopated"]
    idx = rng.randint(0, len(pent) - 1)
    prog = rng.choice(JAZZ_PROGRESSIONS)
    for _ in range(6):
        for chord_ivs in prog:
            midi_chord = [_pc_to_midi(tonic_pc + p, 3) for p in chord_ivs]
            _append_chord(part, midi_chord, 2.0, 66)
            for i in range(8):
                idx = max(0, min(len(pent) - 1, idx + rng.choice([-3, -1, 0, 1, 1, 2])))
                _append_note(part, pent[idx], rhythm[i % len(rhythm)], rng.randint(70, 108))
    return s


def generate_pentatonic_ostinato(seed: int, tonic: str, bpm: int) -> stream.Score:
    rng = random.Random(seed)
    tonic_pc = ROOT_PCS[tonic]
    k = key.Key(tonic)
    s, part = _new_score(bpm, "4/4", k, instrument.Flute())
    pent = _scale_pitches(tonic_pc, "pentatonic", (4, 6))
    ostinato = [pent[i % len(pent)] for i in [0, 2, 4, 2, 1, 3, 4, 0]]
    for bar in range(16):
        shift = rng.choice([0, 0, 12, -12])
        for i, p in enumerate(ostinato):
            qlen = 0.5 if i % 3 else 1.0
            _append_note(part, p + shift, qlen, rng.randint(60, 95))
        if bar % 2 == 0:
            _append_chord(part, [pent[0], pent[2], pent[4]], 1.0, 58)
    return s


def generate_counterpoint_like(seed: int, tonic: str, scale_name: str, bpm: int) -> stream.Score:
    rng = random.Random(seed)
    tonic_pc = ROOT_PCS[tonic]
    mode = "minor" if scale_name == "minor" else "major"
    k = key.Key(tonic, mode)
    s, part = _new_score(bpm, "4/4", k, instrument.Violin())
    pitches = _scale_pitches(tonic_pc, scale_name, (4, 5))
    lower = _scale_pitches(tonic_pc, scale_name, (3, 4))
    a, b = 0, 3
    for i in range(80):
        a = max(0, min(len(pitches) - 1, a + rng.choice([-1, 1])))
        b = max(0, min(len(lower) - 1, b + rng.choice([-1, 0, 1])))
        qlen = 0.5 if i % 5 else 1.0
        _append_note(part, pitches[a], qlen, 88)
        if i % 2 == 0:
            _append_note(part, lower[b], qlen, 70)
        if i % 10 == 0:
            _append_chord(
                part,
                [lower[b], pitches[a], pitches[min(len(pitches) - 1, a + 2)]],
                1.0,
                60,
            )
    return s


PIECES = [
    ("classical_c_major", lambda: generate_classical_melody(1, "C", "major", 96)),
    ("classical_g_major", lambda: generate_classical_melody(2, "G", "major", 108)),
    ("classical_d_minor", lambda: generate_classical_melody(3, "D", "minor", 84)),
    ("classical_a_minor", lambda: generate_classical_melody(4, "A", "minor", 88)),
    ("etude_f_major", lambda: generate_arpeggio_etude(5, "F", "major", 100)),
    ("etude_e_minor", lambda: generate_arpeggio_etude(6, "E", "minor", 92)),
    ("etude_bb_major", lambda: generate_arpeggio_etude(7, "Bb", "major", 110)),
    ("waltz_c_major", lambda: generate_waltz(8, "C", "major", 138)),
    ("waltz_a_minor", lambda: generate_waltz(9, "A", "minor", 126)),
    ("waltz_g_minor", lambda: generate_waltz(10, "G", "minor", 132)),
    ("blues_c", lambda: generate_jazz_blues(11, "C", 112)),
    ("blues_g", lambda: generate_jazz_blues(12, "G", 118)),
    ("blues_f", lambda: generate_jazz_blues(13, "F", 100)),
    ("jazz_c_ii_v_i", lambda: generate_jazz_ii_v_i(14, "C", 120)),
    ("jazz_d_ii_v_i", lambda: generate_jazz_ii_v_i(15, "D", 128)),
    ("jazz_a_ii_v_i", lambda: generate_jazz_ii_v_i(16, "A", 108)),
    ("penta_c", lambda: generate_pentatonic_ostinato(17, "C", 90)),
    ("penta_g", lambda: generate_pentatonic_ostinato(18, "G", 102)),
    ("penta_e", lambda: generate_pentatonic_ostinato(19, "E", 94)),
    ("counter_c_major", lambda: generate_counterpoint_like(20, "C", "major", 80)),
    ("counter_d_minor", lambda: generate_counterpoint_like(21, "D", "minor", 76)),
    ("classical_e_pent", lambda: generate_classical_melody(22, "E", "pentatonic", 104)),
    ("classical_f_blues", lambda: generate_classical_melody(23, "F", "blues", 98)),
    ("jazz_bb", lambda: generate_jazz_ii_v_i(24, "Bb", 116)),
]


def generate_dataset(output_dir: Path | None = None) -> list[Path]:
    out = output_dir or MIDI_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, factory in PIECES:
        score = factory()
        path = out / f"{name}.mid"
        score.write("midi", fp=str(path))
        written.append(path)
        print(f"Wrote {path.name}")
    print(f"Generated {len(written)} MIDI files in {out}")
    return written


if __name__ == "__main__":
    generate_dataset()

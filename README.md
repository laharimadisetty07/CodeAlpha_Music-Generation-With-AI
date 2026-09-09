# AI Music Generation using LSTM & MIDI

An end-to-end deep learning project that algorithmically generates synthetic MIDI music sequences, parses note/chord patterns using `music21`, and trains a Recurrent Neural Network (LSTM) to generate brand-new, original music compositions.

---

## Key Features

* **Algorithmic MIDI Generator:** Built-in script using `music21` to generate clean, structured synthetic MIDI training data across multiple scales and rhythms.
* **Sequence Preprocessing:** Converts MIDI streams into single notes and pitch-cluster chords, mapped to integer vectors using sliding window sequences.
* **Deep LSTM Architecture:** Multi-layer Long Short-Term Memory network trained to capture temporal dependencies and melodic progression in musical sequences.
* **MIDI Exporter:** Generates autoregressive predictions from seed prompts and renders output sequences back into play-ready `.mid` files.

---

## Project Structure

```text
├── data/
│   ├── midi_files/           # Directory containing training .mid files
│   └── generate_midi.py      # Synthetic MIDI dataset generator
├── src/
│   ├── preprocess.py         # Parses MIDI into notes, chords & sequences
│   └── model.py              # PyTorch/TensorFlow LSTM Network Architecture
├── checkpoints/              # Saved model weights & vocab mappings
├── output/                   # Directory where generated .mid files are saved
├── train.py                  # Model training script
├── generate.py               # Music generation & inference script
├── requirements.txt          # Required Python libraries
└── README.md                 # Project Documentation

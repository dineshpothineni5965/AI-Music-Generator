# AI Music Generator Using LSTM Neural Networks

## Project Overview

This project generates new musical sequences using a Long Short-Term Memory (LSTM) neural network.

The system learns patterns from MIDI music and predicts the next musical event based on the previous 50 events. The predicted events are then converted back into a MIDI file that can be played using a MIDI synthesizer.

## Problem Statement

Music contains sequential patterns such as notes, chords, and melodies. Traditional rule-based systems require manually designed musical rules.

The objective of this project is to use machine learning to learn musical patterns automatically and generate new music sequences.

## Objectives

- Process MIDI music data.
- Extract notes and chords from MIDI files.
- Convert musical events into numerical sequences.
- Train an LSTM neural network to predict the next musical event.
- Generate new musical sequences using the trained model.
- Save the generated music as a MIDI file.
- Evaluate the model using training and validation loss and accuracy.

## Dataset

The project uses the MAESTRO v3.0.0 MIDI dataset from Google Magenta.

Dataset:
https://magenta.tensorflow.org/datasets/maestro

The complete MIDI dataset contains 1,276 MIDI files.

The MIDI files were processed using the `music21` library.

### Dataset Processing

For each MIDI file:

1. The MIDI file is loaded using `music21`.
2. Notes are extracted as pitch names.
3. Chords are represented using pitch-class combinations.
4. The extracted events are stored as a sequence.
5. A vocabulary is created by assigning an integer ID to each unique musical event.

The final processed dataset contains:

- 3,696,732 musical events
- 3,158 unique musical events
- Sequence length: 50 events

## Methodology

The project follows this pipeline:

MIDI Dataset
→ MIDI Preprocessing
→ Note/Chord Extraction
→ Integer Encoding
→ Sequence Creation
→ LSTM Training
→ Next Event Prediction
→ MIDI Generation

## Model Architecture

The neural network uses the following architecture:

- Embedding Layer: 128 dimensions
- LSTM Layer: 128 units
- Dropout: 0.2
- Dense Layer: vocabulary size
- Output Activation: Softmax

### Training Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 128
- Training Samples: 300,000
- Validation Samples: 30,000
- Epochs: 8

A representative subset of the prepared sequences was used for practical CPU-based training.

## Music Generation

The trained model receives an initial sequence of 50 musical events.

It then repeatedly:

1. Predicts the next musical event.
2. Samples an event using temperature-based sampling.
3. Adds the predicted event to the sequence.
4. Removes the oldest event.
5. Uses the updated sequence for the next prediction.

Temperature used for generation:

`0.8`

The generated sequence contains 350 events including the initial 50-event seed.

The final output is saved as:

`output/generated_music.mid`

## Results

The model showed continuous improvement during training.

Training loss:

`5.3483 → 4.0902`

Validation loss:

`4.9718 → 4.2490`

Training accuracy:

`2.13% → 13.00%`

Validation accuracy:

`4.34% → 12.44%`

The decreasing loss and increasing accuracy indicate that the LSTM learned meaningful sequential patterns from the musical data.

Because the task contains 3,158 possible musical events, exact next-event accuracy is a difficult metric. Therefore, the generated MIDI output is also used as a qualitative demonstration.

## Project Structure

```text
AI-Music-Generator/
│
├── dataset/
│   └── processed/
│       ├── int_to_note.pkl
│       └── note_to_int.pkl
│
├── models/
│   ├── best_music_generator.keras
│   └── music_generator.keras
│
├── output/
│   ├── accuracy_curve.png
│   ├── generated_music.mid
│   └── loss_curve.png
│
├── src/
│   ├── analyze_data.py
│   ├── generate.py
│   ├── plot_training.py
│   ├── prepare_sequences.py
│   ├── preprocess.py
│   ├── test_midi.py
│   └── train.py
│
├── .gitignore
├── play_music.sh
├── requirements.txt
└── README.md

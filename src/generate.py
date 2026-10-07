import numpy as np
import pickle
import tensorflow as tf
from music21 import stream, note, chord, tempo

MODEL_PATH = "models/music_generator.keras"
SEED_PATH = "dataset/processed/seed.pkl"
NOTE_TO_INT_PATH = "dataset/processed/note_to_int.pkl"
INT_TO_NOTE_PATH = "dataset/processed/int_to_note.pkl"

SEQUENCE_LENGTH = 50
GENERATE_LENGTH = 300
TEMPERATURE = 0.8

model = tf.keras.models.load_model(MODEL_PATH)

with open(SEED_PATH, "rb") as f:
    seed = pickle.load(f)

with open(NOTE_TO_INT_PATH, "rb") as f:
    note_to_int = pickle.load(f)

with open(INT_TO_NOTE_PATH, "rb") as f:
    int_to_note = pickle.load(f)

print("Model loaded successfully!")
print("Vocabulary size:", len(note_to_int))

pattern = list(seed)
generated_notes = list(pattern)

print("Generating music...")

for _ in range(GENERATE_LENGTH):

    encoded = np.array(
        [[note_to_int[n] for n in pattern]],
        dtype=np.int32
    )

    prediction = model.predict(encoded, verbose=0)[0]

    prediction = np.log(prediction + 1e-8) / TEMPERATURE
    probabilities = np.exp(prediction)
    probabilities /= np.sum(probabilities)

    index = np.random.choice(
        len(probabilities),
        p=probabilities
    )

    result = int_to_note[index]

    generated_notes.append(result)

    pattern = pattern[1:]
    pattern.append(result)

print("Generated events:", len(generated_notes))

output = stream.Stream()
output.append(tempo.MetronomeMark(number=100))

for element in generated_notes:

    if "." in element and all(
        part.isdigit() for part in element.split(".")
    ):
        pitches = [int(part) for part in element.split(".")]

        new_chord = chord.Chord(pitches)
        new_chord.quarterLength = 0.5

        output.append(new_chord)

    else:
        try:
            new_note = note.Note(element)
            new_note.quarterLength = 0.5
            output.append(new_note)
        except Exception:
            pass

output.write(
    "midi",
    fp="output/generated_music.mid"
)

print("Music generation complete!")
print("Saved to output/generated_music.mid")

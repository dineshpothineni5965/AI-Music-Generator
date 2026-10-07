import pickle
import numpy as np
from pathlib import Path

SEQUENCE_LENGTH = 50

with open("dataset/processed/notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitchnames)}

num_samples = len(notes) - SEQUENCE_LENGTH

network_input = np.empty((num_samples, SEQUENCE_LENGTH), dtype=np.int32)
network_output = np.empty(num_samples, dtype=np.int32)

for i in range(num_samples):
    sequence = notes[i:i + SEQUENCE_LENGTH]
    target = notes[i + SEQUENCE_LENGTH]

    network_input[i] = [note_to_int[n] for n in sequence]
    network_output[i] = note_to_int[target]

Path("dataset/processed").mkdir(parents=True, exist_ok=True)

np.save("dataset/processed/network_input.npy", network_input)
np.save("dataset/processed/network_output.npy", network_output)

with open("dataset/processed/note_to_int.pkl", "wb") as f:
    pickle.dump(note_to_int, f)

with open("dataset/processed/int_to_note.pkl", "wb") as f:
    pickle.dump({v: k for k, v in note_to_int.items()}, f)

print("Sequence preparation complete!")
print("Total events:", len(notes))
print("Vocabulary size:", len(pitchnames))
print("Sequence length:", SEQUENCE_LENGTH)
print("Training samples:", len(network_input))
print("Input shape:", network_input.shape)
print("Output shape:", network_output.shape)
print("Input memory:", round(network_input.nbytes / (1024 ** 2), 2), "MB")

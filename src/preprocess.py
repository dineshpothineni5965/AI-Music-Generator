import os
import pickle
from music21 import converter, note, chord

DATASET_PATH = "dataset/midi/maestro-v3.0.0"
OUTPUT_PATH = "dataset/processed"

notes = []
processed = 0
max_files = 1276

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.endswith((".mid", ".midi")):
            continue

        file_path = os.path.join(root, file)

        try:
            midi = converter.parse(file_path)

            for element in midi.flatten().notes:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

            processed += 1
            print(f"Processed {processed}/{max_files}")

            if processed >= max_files:
                break

        except Exception as e:
            print("Skipped:", file_path)

    if processed >= max_files:
        break

os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(os.path.join(OUTPUT_PATH, "notes.pkl"), "wb") as f:
    pickle.dump(notes, f)

print("\nPreprocessing complete!")
print("Files processed:", processed)
print("Musical events:", len(notes))
print("Saved to:", OUTPUT_PATH + "/notes.pkl")

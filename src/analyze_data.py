import pickle
from collections import Counter

with open("dataset/processed/notes.pkl", "rb") as f:
    notes = pickle.load(f)

counter = Counter(notes)

print("Total musical events:", len(notes))
print("Unique musical events:", len(counter))

print("\nMost common events:")

for event, count in counter.most_common(10):
    print(event, ":", count)

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

INPUT_PATH = "dataset/processed/network_input.npy"
OUTPUT_PATH = "dataset/processed/network_output.npy"
MAPPING_PATH = "dataset/processed/note_to_int.pkl"

TRAIN_SAMPLES = 300000
BATCH_SIZE = 128
EPOCHS = 8

X = np.load(INPUT_PATH, mmap_mode="r")
y = np.load(OUTPUT_PATH, mmap_mode="r")

with open(MAPPING_PATH, "rb") as f:
    note_to_int = pickle.load(f)

vocab_size = len(note_to_int)
total_samples = len(X)

print("Full dataset samples:", total_samples)
print("Vocabulary size:", vocab_size)

if TRAIN_SAMPLES > total_samples:
    TRAIN_SAMPLES = total_samples

validation_size = int(TRAIN_SAMPLES * 0.1)
train_size = TRAIN_SAMPLES - validation_size

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:TRAIN_SAMPLES]
y_val = y[train_size:TRAIN_SAMPLES]

class MusicSequence(tf.keras.utils.Sequence):

    def __init__(self, X, y, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.X))

        batch_indices = self.indices[start:end]
        batch_indices = np.sort(batch_indices)

        X_batch = np.asarray(self.X[batch_indices], dtype=np.int32)
        y_batch = np.asarray(self.y[batch_indices], dtype=np.int32)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_data = MusicSequence(
    X_train,
    y_train,
    BATCH_SIZE
)

validation_data = MusicSequence(
    X_val,
    y_val,
    BATCH_SIZE,
    shuffle=False
)

model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(128),
    Dropout(0.2),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.build((None, 50))

model.summary()

checkpoint = ModelCheckpoint(
    "models/best_music_generator.keras",
    monitor="val_loss",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

model.save("models/music_generator.keras")

print("Training complete!")
print("Model saved to models/music_generator.keras")

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import *

# =====================================
# Configuration
# =====================================

DATASET = "input.txt"

WINDOW_SIZE = 20

MAX_WORDS = 10000

EMBEDDING_DIM = 64

LSTM_UNITS = 64

BATCH_SIZE = 256

EPOCHS = 10

MODEL_FILE = "text_generator.keras"

TOKENIZER_FILE = "tokenizer.pkl"

# =====================================
# Load Dataset
# =====================================

print("Loading Dataset...")

with open(DATASET, "r", encoding="utf-8") as f:
    text = f.read()

text = clean_text(text)

# Use only first 50,000 words
words = text.split()[:50000]

text = " ".join(words)

print("Words :", len(words))

# =====================================
# Tokenizer
# =====================================

tokenizer = create_tokenizer(text)

save_tokenizer(tokenizer)

total_words = min(MAX_WORDS, len(tokenizer.word_index) + 1)

print("Vocabulary :", total_words)

# =====================================
# Build Training Data
# =====================================

print("Preparing Sequences...")

input_sequences = []

for i in range(WINDOW_SIZE, len(words)):

    seq = words[i-WINDOW_SIZE:i+1]

    encoded = tokenizer.texts_to_sequences(
        [" ".join(seq)]
    )[0]

    if len(encoded) == WINDOW_SIZE + 1:
        input_sequences.append(encoded)

input_sequences = np.array(input_sequences)

print("Training Samples :", len(input_sequences))

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(
    y,
    num_classes=total_words
)

# =====================================
# Build Model
# =====================================

print("Building Model...")

model = Sequential()

model.add(
    Embedding(
        input_dim=total_words,
        output_dim=EMBEDDING_DIM,
        input_length=WINDOW_SIZE
    )
)

model.add(
    LSTM(
        LSTM_UNITS
    )
)

model.add(
    Dropout(
        0.2
    )
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dense(
        total_words,
        activation="softmax"
    )
)

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# =====================================
# Callbacks
# =====================================

callbacks = [

    EarlyStopping(

        monitor="loss",

        patience=2,

        restore_best_weights=True

    ),

    ModelCheckpoint(

        MODEL_FILE,

        save_best_only=True

    )

]

# =====================================
# Train
# =====================================

print("\nTraining Started...\n")

history = model.fit(

    X,

    y,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1

)

print("\nTraining Finished!")

model.save(MODEL_FILE)

print("Model Saved Successfully!")
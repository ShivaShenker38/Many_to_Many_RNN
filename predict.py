import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import load_tokenizer, clean_text

# =====================================
# Configuration
# =====================================

MODEL_FILE = "text_generator.keras"
WINDOW_SIZE = 20

# =====================================
# Load Model
# =====================================

print("Loading Model...")

model = load_model(MODEL_FILE)

print("Model Loaded Successfully!")

# =====================================
# Load Tokenizer
# =====================================

tokenizer = load_tokenizer()

# =====================================
# Temperature Sampling
# =====================================

def sample(preds, temperature=0.8):

    preds = np.asarray(preds).astype("float64")

    preds = np.log(preds + 1e-8) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

# =====================================
# Generate Text
# =====================================

def generate_text(seed_text,
                  next_words=100,
                  temperature=0.8):

    seed_text = clean_text(seed_text)

    output = seed_text

    for _ in range(next_words):

        sequence = tokenizer.texts_to_sequences([output])[0]

        sequence = sequence[-WINDOW_SIZE:]

        sequence = pad_sequences(
            [sequence],
            maxlen=WINDOW_SIZE,
            padding="pre"
        )

        prediction = model.predict(
            sequence,
            verbose=0
        )[0]

        predicted_index = sample(
            prediction,
            temperature
        )

        word = tokenizer.index_word.get(predicted_index)

        if word is None:
            continue

        output += " " + word

    return output

# =====================================
# Command Line Testing
# =====================================

if __name__ == "__main__":

    while True:

        seed = input("\nEnter Seed Text (or 'exit'): ")

        if seed.lower() == "exit":
            break

        generated = generate_text(
            seed_text=seed,
            next_words=100,
            temperature=0.8
        )

        print("\nGenerated Text:\n")
        print(generated)
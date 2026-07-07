import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Configuration
# ==============================

MAX_WORDS = 10000
WINDOW_SIZE = 20

# ==============================
# Clean Text
# ==============================

def clean_text(text):
    text = text.lower()

    # Keep letters, numbers and punctuation
    text = re.sub(r"[^a-z0-9.,!?;:'\"\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==============================
# Create Tokenizer
# ==============================

def create_tokenizer(text):

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts([text])

    return tokenizer

# ==============================
# Save Tokenizer
# ==============================

def save_tokenizer(tokenizer):

    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

# ==============================
# Load Tokenizer
# ==============================

def load_tokenizer():

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer

# ==============================
# Convert Text to Sequence
# ==============================

def text_to_sequence(tokenizer, text):

    sequence = tokenizer.texts_to_sequences([text])

    sequence = pad_sequences(
        sequence,
        maxlen=WINDOW_SIZE,
        padding="pre"
    )

    return sequence
import os, warnings, logging
# Remove envrionment warnings from terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = only errors
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import re
import csv

# CONSTANTS

tf.config.set_visible_devices([], 'GPU')
    # Common simple animal labels — expanded and grouped by type
ANIMAL_KEYWORDS: dict[str:list[str]] = {
    "dog": ["dog", "canine", "canidae", "bark", "bow-wow"],
    "cat": ["cat", "feline", "meow", "purr"],
    "bird": ["bird", "chirp", "tweet", "singing bird", "crow", "rooster", "hen", "duck"],
    "cow": ["cow", "moo", "cattle"],
    "sheep": ["sheep", "bleat"],
    "pig": ["pig", "oink"],
    "horse": ["horse", "neigh", "whinny"],
    "bear": ["bear", "growl"],
    "wolf": ["wolf", "howl", "canidae", "wild dog"],
    "fox": ["fox", "vulpes"],
    "deer": ["deer", "stag"],
    "elk": ["elk"],
    "lion": ["lion", "roar"],
    "tiger": ["tiger", "roar"]
}
    #Model to use, replace URL with your model (results may vary)
MODEL: str = "https://tfhub.dev/google/yamnet/1"


# Load (YAMNet) model
yamnet_model = hub.load(MODEL)

# Load labels
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)

# Parse labels
with open(class_map_path, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    class_names: list = [row[2] for row in reader]

def classify_sound(file_path: str, top_k: int=5) -> tuple:
    # Load audio
    waveform, _sr = librosa.load(file_path, sr=16000, mono=True)
    
    # Run model
    scores, _embeddings, _spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    # Get top predictions
    top_indices = np.argsort(mean_scores)[::-1][:top_k]
    top_labels = [(class_names[i], float(mean_scores[i])) for i in top_indices]

    # Return the top label and full list
    best_label, best_conf = top_labels[0]
    return best_label, best_conf, top_labels


def simplify_label(top_labels: list[tuple])->tuple:
    """
    Takes a list of (label, confidence) tuples from YAMNet and returns
    the best-matching simple animal name with strict word boundaries.
    """
    best_match = None
    best_conf = 0.0

    for label, conf in top_labels:
        label_lower = label.lower()
        for animal, keywords in ANIMAL_KEYWORDS.items():
            for k in keywords:
                # Match full words or comma/space boundaries only
                pattern = rf"\b{k}\b"
                if re.search(pattern, label_lower):
                    if conf > best_conf:
                        best_match = animal.capitalize()
                        best_conf = conf

    # Fallback if nothing matches
    if not best_match:
        top_label, top_conf = top_labels[0]
        best_match = top_label.split(",")[0].split(" ")[0].capitalize()
        best_conf = top_conf

    return best_match, best_conf



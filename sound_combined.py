# Disable excessive logs
import os, warnings, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import torch
import csv
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


#############################################
#       Lazy-loaded global models           #
#############################################

YAMNET_MODEL = None
YAMNET_LABELS = None

AST_EXTRACTOR = None
AST_MODEL = None


#############################################
#              Load YAMNet                  #
#############################################

def load_yamnet():
    global YAMNET_MODEL, YAMNET_LABELS

    if YAMNET_MODEL is not None:
        return  # already loaded

    print("Loading YAMNet model...")
    YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")

    yamnet_csv = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    )

    with open(yamnet_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        YAMNET_LABELS = [row[2] for row in reader]


def classify_yamnet(file_path, top_k=5):
    load_yamnet()

    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    scores, embeddings, spectrogram = YAMNET_MODEL(waveform)
    scores = scores.numpy()
    mean_scores = np.mean(scores, axis=0)

    top_ids = np.argsort(mean_scores)[::-1][:top_k]
    top_labels = [(YAMNET_LABELS[i], float(mean_scores[i])) for i in top_ids]

    best_label, best_conf = top_labels[0]
    return best_label, best_conf, top_labels


#############################################
#                Load AST                   #
#############################################

def load_ast():
    global AST_EXTRACTOR, AST_MODEL

    if AST_MODEL is not None:
        return  # already loaded

    print("Loading AST model...")
    AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

    AST_EXTRACTOR = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
    AST_MODEL = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME)


def classify_ast(file_path):
    load_ast()

    waveform, sr = librosa.load(file_path, sr=AST_EXTRACTOR.sampling_rate, mono=True)
    inputs = AST_EXTRACTOR(waveform, sampling_rate=AST_EXTRACTOR.sampling_rate, return_tensors="pt")

    with torch.no_grad():
        logits = AST_MODEL(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        conf = torch.softmax(logits, dim=-1)[0][pred_id].item()
        label = AST_MODEL.config.id2label[pred_id]

    return label, conf


#############################################
#       Animal Classification Keywords       #
#############################################

ANIMAL_MAP = {
    "wolf": ["wolf", "wolves", "howl", "canidae", "canine", "wild dog"],
    "dog": ["dog", "bark", "bow-wow", "canine", "puppy", "domestic animals, pets"],
    "cat": ["cat", "feline", "meow", "purr"],
    "bear": ["bear", "growl", "grizzly", "ursidae"],
    "fox": ["fox", "vulpes"],
    "cow": ["cow", "moo", "cattle"],
    "sheep": ["sheep", "bleat"],
    "pig": ["pig", "oink", "hog"],
    "horse": ["horse", "whinny", "neigh"],
    "deer": ["deer", "stag"],
    "elk": ["elk"],
    "lion": ["lion", "big cat", "panthera leo", "roar"],
    "tiger": ["tiger", "roar"],
    "bird": ["bird", "chirp", "tweet", "songbird", "singing bird", "rooster", "hen", "duck", "crow"],
    "snake": ["hiss", "snake", "serpent", "rattle", "rattlesnake"]
}


def classify_animal(yamnet_top_labels, ast_label=None, ast_conf=0.0):
    scores = {animal: 0.0 for animal in ANIMAL_MAP.keys()}

    # Score YAMNet
    for label, conf in yamnet_top_labels:
        label_lower = label.lower()
        for animal, keywords in ANIMAL_MAP.items():
            if any(k in label_lower for k in keywords):
                scores[animal] += conf

    # Score AST
    if ast_label:
        ast_lower = ast_label.lower()
        for animal, keywords in ANIMAL_MAP.items():
            if any(k in ast_lower for k in keywords):
                scores[animal] += ast_conf

    # Pick one best animal
    best_animal = max(scores, key=lambda a: scores[a])
    best_score = scores[best_animal]

    if best_score == 0:
        fallback = yamnet_top_labels[0][0].split(",")[0].split(" ")[0]
        return fallback.capitalize(), 0.0

    normalized_conf = min(1.0, round(best_score / 2, 2))
    return best_animal.capitalize(), normalized_conf



#############################################
#             Combined Logic                #
#############################################

def classify_combined(file_path):
    y_label, y_conf, y_all = classify_yamnet(file_path)
    a_label, a_conf = classify_ast(file_path)

    if y_label.lower() in a_label.lower() or a_label.lower() in y_label.lower():
        final_label = y_label if y_conf >= a_conf else a_label
        final_conf = max(y_conf, a_conf)
    else:
        final_label, final_conf = (a_label, a_conf) if a_conf - y_conf > 0.1 else (y_label, y_conf)

    animal, animal_score = classify_animal(y_all, a_label, a_conf)

    return {
        "yamnet": (y_label, y_conf),
        "ast": (a_label, a_conf),
        "final": (final_label, final_conf),
        "animal": (animal, animal_score),
        "yamnet_full": y_all
    }


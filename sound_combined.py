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
import re
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

#############################################
#                YAMNet                     #
#############################################

print("Loading YAMNet...https://tfhub.dev/google/yamnet/1")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

yamnet_csv = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)

with open(yamnet_csv, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    yamnet_labels = [row[2] for row in reader]

def classify_yamnet(file_path, top_k=5):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores = scores.numpy()
    mean_scores = np.mean(scores, axis=0)

    top_ids = np.argsort(mean_scores)[::-1][:top_k]
    top_labels = [(yamnet_labels[i], float(mean_scores[i])) for i in top_ids]

    best_label, best_conf = top_labels[0]
    return best_label, best_conf, top_labels


#############################################
#                AST Model                  #
#############################################

print("Loading AST model...MIT/ast-finetuned-audioset-10-10-0.4593")
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
ast_model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME)

def classify_ast(file_path):
    waveform, sr = librosa.load(file_path, sr=ast_extractor.sampling_rate, mono=True)
    inputs = ast_extractor(waveform, sampling_rate=ast_extractor.sampling_rate, return_tensors="pt")

    with torch.no_grad():
        logits = ast_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        conf = torch.softmax(logits, dim=-1)[0][pred_id].item()
        label = ast_model.config.id2label[pred_id]

    return label, conf


#############################################
#            Animal Classification          #
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

    # Score YAMNet top labels
    for label, conf in yamnet_top_labels:
        label_lower = label.lower()
        for animal, keywords in ANIMAL_MAP.items():
            for k in keywords:
                if k in label_lower:
                    scores[animal] += conf

    # Score AST label
    if ast_label:
        ast_lower = ast_label.lower()
        for animal, keywords in ANIMAL_MAP.items():
            for k in keywords:
                if k in ast_lower:
                    scores[animal] += ast_conf

    # Pick animal with highest raw score
    best_animal = max(scores, key=lambda a: scores[a])
    best_raw_score = scores[best_animal]

    # Normalize confidence: average YAMNet + AST contributions
    normalized_conf = min(1.0, round(best_raw_score / 2, 2))

    # Fallback
    if best_raw_score == 0:
        fallback = yamnet_top_labels[0][0].split(",")[0].split(" ")[0]
        return fallback.capitalize(), 0.0

    return best_animal.capitalize(), normalized_conf



#############################################
#             Combined Logic                #
#############################################

def classify_combined(file_path):
    y_label, y_conf, y_all = classify_yamnet(file_path)
    a_label, a_conf = classify_ast(file_path)

    # Determine combined ML label
    if y_label.lower() in a_label.lower() or a_label.lower() in y_label.lower():
        final_label = y_label if y_conf >= a_conf else a_label
        final_conf = max(y_conf, a_conf)
    else:
        if a_conf - y_conf > 0.1:
            final_label, final_conf = a_label, a_conf
        else:
            final_label, final_conf = y_label, y_conf

    # NEW: classify animal
    animal, animal_score = classify_animal(y_all, a_label, a_conf)

    return {
        "yamnet": (y_label, y_conf),
        "ast": (a_label, a_conf),
        "final": (final_label, final_conf),
        "animal": (animal, animal_score),
        "yamnet_full": y_all
    }


#############################################
#                 TESTING                   #
#############################################

if __name__ == "__main__":
    test_file = "data/deer.mp3"

    result = classify_combined(test_file)

    print("\n================ RESULTS ================\n")
    print(f"YAMNet → {result['yamnet'][0]} ({result['yamnet'][1]:.2f})")
    print(f"AST    → {result['ast'][0]} ({result['ast'][1]:.2f})")
    print(f"Combined ML → {result['final'][0]} ({result['final'][1]:.2f})")
    print(f"\nANIMAL → {result['animal'][0]} ({result['animal'][1]:.2f})")
    print("\n=========================================\n")

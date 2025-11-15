from sound_classification import classify_sound, simplify_label

# Analyze your sound
best_label, best_conf, top_labels = classify_sound("data/fox.mp3")

# Simplify label
simple_label, confidence = simplify_label(top_labels)

print(f"Predicted: {simple_label} ({confidence:.2f} confidence)\n")

print("Top 5 Raw Predictions:")
for label, score in top_labels:
    print(f"  {label}: {score:.2f}")

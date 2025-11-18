from sound_classification import classify_sound, simplify_label
from describe import describe_animal

if __name__=='__main__':
    best_label, best_conf, top_labels = classify_sound("data/lion.mp3")

    best_match, best_conf = simplify_label(top_labels=top_labels)

    description = describe_animal(best_match)

    print(description)
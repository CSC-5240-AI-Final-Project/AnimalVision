from src.sound_classification import classify_sound, simplify_label
from src.describe import describe_animal
from src.image_gen import generate_image

if __name__=='__main__':
    best_label, best_conf, top_labels = classify_sound("data/lion.mp3")

    best_match, best_conf = simplify_label(top_labels=top_labels)

    description = describe_animal(best_match)
    
    image = generate_image(best_match)

    image.save('data')
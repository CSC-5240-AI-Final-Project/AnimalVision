from src.sound_classification import classify_sound, simplify_label
from src.describe import describe_animal
from src.image_gen import generate_image

if __name__=='__main__':
    print('Getting best label...')
    best_label, best_conf, top_labels = classify_sound("data/lion.mp3")

    print("Simplifying label...")
    best_match, best_conf = simplify_label(top_labels=top_labels)

    print("Getting description...")
    description = describe_animal(best_match)
    
    print("Generating image...")
    image = generate_image(best_match)

    print("Saving image...")
    image.save(f'{best_match}.png')
    print(f'Saved as {best_match}.png')
import torch
from diffusers import DiffusionPipeline

# switch to "mps" for apple devices

def generate_image(animal: str, model: str ="stabilityai/stable-diffusion-xl-base-1.0"):
    pipe = DiffusionPipeline.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda")
    prompt = f"Photo realistic depiction of {animal}"
    image = pipe(prompt).images[0]
    return image



import torch
from diffusers import DiffusionPipeline

# Global cached pipeline
PIPE = None


# Detect best available device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Lazy load the SDXL-Turbo model
def load_pipeline(model_name: str = "stabilityai/sdxl-turbo"):
    global PIPE
    if PIPE is not None:
        return PIPE

    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading SDXL-Turbo pipeline...")

    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype
    ).to(device)

    PIPE = pipe
    return PIPE


def generate_image(animal: str):
    """
    Generates a high-quality AND fast image using SDXL-Turbo.
    """
    pipe = load_pipeline()

    # SDXL-Turbo typically uses ~2–4 steps
    prompt = f"Ultra-detailed, photorealistic depiction of a {animal}, natural lighting, high detail, 85mm lens, shallow depth of field"

    image = pipe(
        prompt,
        num_inference_steps=2,      # Fast + good quality
        guidance_scale=0.0          # Turbo models work best with zero guidance
    ).images[0]

    return image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wikipedia as wp

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = None
model = None


#############################################################
#                Load model ONCE (no pipeline)              #
#############################################################
def load_llm():
    global tokenizer, model
    if model is not None:
        return

    print("Loading TinyLlama 1.1B model (optimized)...")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=dtype,
    ).to(device)

    model.eval()


#############################################################
#               Article Trimming (faster)                   #
#############################################################
def truncate_article(article: str, max_words=300):
    words = article.split()
    if len(words) > max_words:
        words = words[:max_words]
    text = " ".join(words)
    # Try to end on sentence
    if "." in text:
        text = text.rsplit(".", 1)[0] + "."
    return text


#############################################################
#            MAIN: Generate Animal Description              #
#############################################################
def describe_animal(
    animal_name: str,
    prompt: str = "prompts/describe_animal.md",
    max_new_tokens=120
) -> str:

    load_llm()

    # Load system instructions
    with open(prompt, "r") as f:
        system_prompt = f.read().strip()

    # Retrieve Wikipedia
    try:
        article_raw = wp.page(animal_name, auto_suggest=False).content
    except:
        article_raw = f"No usable article found for {animal_name}."

    article = truncate_article(article_raw)

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Animal: {animal_name}\n\n"
        f"Facts:\n{article}\n\n"
        f"Write the final description below:\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt from the output
    if text.startswith(full_prompt):
        text = text[len(full_prompt):].strip()

    return text

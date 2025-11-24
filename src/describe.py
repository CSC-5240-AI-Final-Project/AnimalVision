from src.sound_classification import classify_sound, simplify_label
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import wikipedia as wp

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def describe_animal(simple_label: str, prompt: str='prompts/describe_animal.md', model: str="meta-llama/Llama-3.1-8B-Instruct", temp: float =0.3, top: float = 0.9) -> str:
    
    '''
    prompt = Path to prompt file (we use .md)
    sound = Path to sound file
    model = path to model 
    temp = temperature used by model (keep 0-1)
    top = top_p (keep 0-1)
    '''

    #Model setup
    model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model)
    generator = pipeline('text-generation', model=model,tokenizer=tokenizer, return_full_text=False, 
                         do_sample=True, temperature=temp, top_p=top)

    #Prompt setup
    with open(prompt) as f:
        prompt = f.read()

    #Titling for wikipedia parsing
    title = simple_label.strip().title()

    #We do assume an animal has a page associated with its name :3
    article = wp.page(title, auto_suggest=False, redirect=True) #Getting artile
    article = article.content #Full article content

    #Generate Output
    prompts = [{'role':'system', 'content':f'{prompt}'}, {"role":'user', 'content':f'Animal: {simple_label} Article: {article}'}]

    output = generator(prompts, truncation=True)

    return simple_label, confidence, output[0]['generated_text'] 
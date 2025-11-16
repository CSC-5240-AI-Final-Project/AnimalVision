from sound_classification import classify_sound, simplify_label
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#Model setup
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL)

gener = pipeline('text-generation', model=model,tokenizer=tokenizer, return_full_text=False, do_sample=True, temperature=.3, top_p=.9)

#Prompt setup
with open('prompts/describe_animal.md') as f:
    prompt = f.read()

# Analyze sound
best_label, best_conf, top_labels = classify_sound("data/lion.mp3")

# Simplify label
simple_label, confidence = simplify_label(top_labels)


#Generate Output
prompts = [{'role':'system', 'content':f'{prompt}'}, {"role":'user', 'content':f'Animal{simple_label}'}]

output = gener(prompts, truncation=True)

print(output[0]['generated_text'])
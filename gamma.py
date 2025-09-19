from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "google/gemma-3-4b-it"  # This is the actual available ID

# Load tokenizer and model on CPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"})

# Create CPU-based pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# Prompt
prompt = "Explain quantum computing in simple terms."

# Run inference
output = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
print(output[0]['generated_text'])

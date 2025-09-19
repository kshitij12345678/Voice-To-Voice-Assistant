from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import gc

# Clear previous GPU memory
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model_id = "google/gemma-3-4b-it"

# 4-bit quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load quantized model to GPU or CPU automatically
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# Text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test prompt
prompt = "Explain quantum computing in simple terms."

# Run inference
output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
print(output[0]['generated_text'])

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

model_id = "google/gemma-3-4b-it"

# 8-bit quantization config
bnb_config = BitsAndBytesConfig(
    
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load 8-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Inference pipeline (device_map handles GPU placement)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt
prompt = "Explain quantum computing in simple terms."

# Generate
output = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
print(output[0]['generated_text'])

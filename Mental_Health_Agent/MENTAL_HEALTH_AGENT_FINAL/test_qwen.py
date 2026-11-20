
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model location
MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--Qwen--Qwen2.5-1.5B-Instruct",
    "snapshots",
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

print("=" * 55)
print("QWEN MODEL TEST - MENTAL HEALTH AGENT V2")
print("=" * 55)

# Check files
assert os.path.isfile(os.path.join(MODEL_PATH, "config.json"))
assert os.path.isfile(os.path.join(MODEL_PATH, "model.safetensors"))

print("\n[1] Model files: OK")
print("[2] Device: CPU")
print("[3] Loading tokenizer...")

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded successfully.")

print("\n[4] Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

model.eval()

load_time = time.time() - start_time

print("Model loaded successfully.")
print(f"Loading time: {load_time:.2f} seconds")

# Test question
messages = [
    {
        "role": "system",
        "content": (
            "You are an educational mental health information assistant. "
            "Do not diagnose or prescribe treatment."
        )
    },
    {
        "role": "user",
        "content": "What is stress? Explain in simple English."
    }
]

print("\n[5] Generating response...")

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt")

start_time = time.time()

with torch.inference_mode():
    outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    pad_token_id=tokenizer.eos_token_id
)

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

generation_time = time.time() - start_time

print("\n" + "=" * 55)
print("MODEL RESPONSE")
print("=" * 55)
print(answer)

print("\n" + "=" * 55)
print("TEST RESULTS")
print("=" * 55)
print(f"Model: Qwen2.5-1.5B-Instruct")
print(f"Device: CPU")
print(f"Loading time: {load_time:.2f} seconds")
print(f"Generation time: {generation_time:.2f} seconds")
print("Status: TEST COMPLETED")

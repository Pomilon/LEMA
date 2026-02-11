import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
import os

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

print("Creating GPT-2 Small model...")
model = GPT2LMHeadModel(config)

# Break shared weights
model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

print("Saving to medium_gpt2.safetensors...")
save_file(model.state_dict(), "medium_gpt2.safetensors")

size_bytes = os.path.getsize("medium_gpt2.safetensors")
print(f"Created medium_gpt2.safetensors: {size_bytes / 1024 / 1024:.2f} MB")

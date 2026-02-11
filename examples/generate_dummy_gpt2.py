import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

config = GPT2Config(
    vocab_size=1000,
    n_positions=128,
    n_embd=64,
    n_layer=4,
    n_head=4
)
model = GPT2LMHeadModel(config)

# Break shared weights for safetensors
model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

# Save to safetensors
save_file(model.state_dict(), "dummy_gpt2.safetensors")
print("Created dummy_gpt2.safetensors")

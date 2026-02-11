import torch
import torch.optim as optim
import sys
import os

# Add src to path if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lema.core.gbi import GlobalBinaryIndex
from lema.core.lora import LoRAManager
from lema.models.gpt2 import GPT2Adapter
from lema.engine.trainer import LemaTrainer
from lema.config import LemaConfig, MemoryStrategy

# 1. Configuration matching generate_dummy_gpt2.py
hf_config = {
    "vocab_size": 1000,
    "n_positions": 128,
    "n_embd": 64,
    "n_layer": 4,
    "n_head": 4,
    "layer_norm_epsilon": 1e-5,
    "attn_implementation": "eager"
}

model_path = "dummy_gpt2.safetensors"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# 2. LEMA Configuration
lema_config = LemaConfig(
    model_name_or_path=model_path,
    device=device,
    strategy=MemoryStrategy.STREAMING,
    lora_rank=4,
    learning_rate=0.01
)

# 3. Components
print("Initializing Adapter...")
adapter = GPT2Adapter(hf_config)

print("Initializing GBI...")
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Run examples/generate_dummy_gpt2.py first.")
    sys.exit(1)
gbi = GlobalBinaryIndex(model_path)

print("Initializing LoRA Manager...")
lora_config = {"r": lema_config.lora_rank, "alpha": 8, "target_modules": ["c_attn"]}
lora_manager = LoRAManager(lora_config, device=device)

# 4. Initialize LoRA parameters by pre-scanning
print("Pre-scanning layers to initialize LoRA parameters...")
for layer in adapter.get_layer_metadata():
    if layer['type'] == 'block':
        module = adapter.construct_layer_module(layer['id'], None, lora_manager)
        adapter.release_layer_module(module)

# 5. Setup Optimizer
trainable_params = lora_manager.get_trainable_parameters()
print(f"Found {len(trainable_params)} LoRA parameters.")
optimizer = optim.AdamW(trainable_params, lr=lema_config.learning_rate)

# 6. Initialize Trainer
print("Initializing Trainer...")
trainer = LemaTrainer(
    config=lema_config,
    model_adapter=adapter,
    gbi=gbi,
    lora_manager=lora_manager,
    optimizer=optimizer
)

# 7. Training Step
input_ids = torch.randint(0, 1000, (1, 10)).to(device)

print("Executing Training Step...")
logits, loss = trainer.train_step(input_ids, labels=input_ids)

print(f"Success! Training step completed. Loss: {loss:.4f}")
print("Final Output Logits Shape:", logits.shape)
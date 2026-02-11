import sys
import os
import resource
import torch
import time
from transformers import GPT2Config, GPT2LMHeadModel

# Add src to path if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lema.core.gbi import GlobalBinaryIndex
from lema.models.gpt2 import GPT2Adapter
from lema.engine.trainer import LemaTrainer
from lema.core.lora import LoRAManager
from lema.config import LemaConfig, MemoryStrategy

def get_peak_rss_mb():
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def run_baseline(model_path):
    print("--- Running BASELINE ---")
    start_ram = get_peak_rss_mb()
    
    # 1. Config (Matching dummy_gpt2 if path matches)
    if "dummy" in model_path:
        config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=4, n_head=4)
    else:
        config = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12)
    
    # 2. Instantiate Model
    print("Instantiating Model...")
    model = GPT2LMHeadModel(config)
    
    # 3. Load Weights
    print("Loading Weights...")
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    
    print(f"Model loaded. RAM: {get_peak_rss_mb():.2f} MB")
    
    # 4. Forward Pass
    print("Forward Pass...")
    input_ids = torch.randint(0, config.vocab_size, (1, 64))
    output = model(input_ids)
    
    # 5. Backward Pass
    print("Backward Pass...")
    loss = output.logits.mean()
    loss.backward()
    
    peak_ram = get_peak_rss_mb()
    print(f"Baseline Peak RSS: {peak_ram:.2f} MB")
    return peak_ram

def run_lema(model_path):
    print("--- Running LEMA ---")
    start_ram = get_peak_rss_mb()
    
    # 1. Config
    if "dummy" in model_path:
        hf_config = {"vocab_size": 1000, "n_positions": 128, "n_embd": 64, "n_layer": 4, "n_head": 4, "attn_implementation": "eager"}
    else:
        hf_config = {"vocab_size": 50257, "n_positions": 1024, "n_embd": 768, "n_layer": 12, "n_head": 12, "attn_implementation": "eager"}
    
    lema_config = LemaConfig(model_name_or_path=model_path, device="cpu", strategy=MemoryStrategy.STREAMING)
    
    # 2. Components
    print("Initializing Components...")
    adapter = GPT2Adapter(hf_config)
    gbi = GlobalBinaryIndex(model_path)
    
    # LoRA
    lora_config = {"r": 8, "alpha": 16, "target_modules": ["c_attn"]}
    lora_manager = LoRAManager(lora_config, device="cpu")
    
    # Init LoRA params
    for layer in adapter.get_layer_metadata():
        if layer['type'] == 'block':
            module = adapter.construct_layer_module(layer['id'], None, lora_manager)
            adapter.release_layer_module(module)
            
    optimizer = torch.optim.AdamW(lora_manager.get_trainable_parameters(), lr=1e-4)
    trainer = LemaTrainer(lema_config, adapter, gbi, lora_manager=lora_manager, optimizer=optimizer)
    
    print(f"Components Ready. RAM: {get_peak_rss_mb():.2f} MB")
    
    # 3. Train Step
    print("Training Step...")
    input_ids = torch.randint(0, hf_config["vocab_size"], (1, 64))
    trainer.train_step(input_ids, labels=input_ids)
    
    peak_ram = get_peak_rss_mb()
    print(f"LEMA Peak RSS: {peak_ram:.2f} MB")
    return peak_ram

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python examples/benchmark_runner.py [mode] [model_path]")
        sys.exit(1)
        
    mode = sys.argv[1]
    path = sys.argv[2]
    
    if mode == "baseline":
        run_baseline(path)
    elif mode == "lema":
        run_lema(path)
    else:
        print("Unknown mode")
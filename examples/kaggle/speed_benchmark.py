import torch
import gc
import os
import time
import transformers
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig
from safetensors.torch import save_file

# LEMA Imports
from src.lema.core.gbi import GlobalBinaryIndex
from src.lema.models.llama import LlamaAdapter
from src.lema.engine.trainer import LemaTrainer
from src.lema.core.lora import LoRAManager
from src.lema.config import LemaConfig, MemoryStrategy

print(f"Using Transformers version: {transformers.__version__}")

# --- MODELS TO TEST ---
# Focusing on Llama architectures for speed comparison
MODELS = [
    {
        "name": "TinyLlama 1.1B",
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "path": "tinyllama_1b.safetensors",
        "type": "llama"
    },
    {
        "name": "Llama-2 7B",
        "hf_id": "NousResearch/Llama-2-7b-hf",
        "path": "llama2_7b.safetensors",
        "type": "llama"
    }
]

NUM_STEPS = 20 # Enough to stabilize avg speed

def download_and_convert(model_info):
    print(f"\n--- Preparing {model_info['name']} ---")
    if os.path.exists(model_info['path']):
        print(f"{model_info['path']} already exists.")
        return

    print(f"Downloading {model_info['hf_id']}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_info['hf_id'], 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    
    # Break shared weights if necessary
    if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
         if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
             model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

    print(f"Saving to {model_info['path']}...")
    save_file(model.state_dict(), model_info['path'])
    del model
    gc.collect()

def benchmark_peft_speed(model_info):
    print(f"\n>>> BENCHMARKING PEFT SPEED: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_info['hf_id'],
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        
        # Enable Gradient Checkpointing to save memory
        model.gradient_checkpointing_enable()
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=target_modules,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, 1000, (1, 512)).cuda()
        
        # Warmup
        model(input_ids, labels=input_ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        
        print(f"Running {NUM_STEPS} steps...")
        start_time = time.time()
        for _ in range(NUM_STEPS):
            loss = model(input_ids, labels=input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / NUM_STEPS
        print(f"PEFT Avg Time/Step: {avg_time:.4f}s")
        
        del model
        del optimizer
        torch.cuda.empty_cache()
        return avg_time
        
    except Exception as e:
        print(f"PEFT Benchmark Failed: {e}")
        return float('inf')

def benchmark_lema_speed(model_info):
    print(f"\n>>> BENCHMARKING LEMA SPEED: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    download_and_convert(model_info)
    
    # Enable checkpointing only for large models (e.g. 7B)
    use_gc = "7b" in model_info['hf_id'].lower()
    print(f"Gradient Checkpointing: {use_gc}")
    
    try:
        hf_config = AutoConfig.from_pretrained(model_info['hf_id'])
        hf_config_dict = hf_config.to_dict()
        hf_config_dict["attn_implementation"] = "eager"
        hf_config_dict["torch_dtype"] = "float16"
        
        adapter = LlamaAdapter(hf_config_dict)
        gbi = GlobalBinaryIndex(model_info['path'])
        
        lema_config = LemaConfig(
            model_name_or_path=model_info['path'],
            device="cuda",
            strategy=MemoryStrategy.STREAMING,
            learning_rate=1e-4,
            gradient_checkpointing=use_gc
        )
        
        lora_config = {
            "r": 16, "alpha": 32, 
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
        lora_manager = LoRAManager(lora_config, device="cuda")
        
        # Init params
        for layer in adapter.get_layer_metadata():
            if layer['type'] == 'block':
                module = adapter.construct_layer_module(layer['id'], None, lora_manager)
                adapter.release_layer_module(module)
        
        torch.cuda.empty_cache()
        
        trainable_params = lora_manager.get_trainable_parameters()
        optimizer = torch.optim.AdamW(trainable_params, lr=lema_config.learning_rate)
        
        trainer = LemaTrainer(
            config=lema_config,
            model_adapter=adapter, 
            gbi=gbi, 
            lora_manager=lora_manager, 
            optimizer=optimizer
        )
        
        input_ids = torch.randint(0, 1000, (1, 512)).cuda()
        
        # Warmup
        trainer.train_step(input_ids, labels=input_ids)
        torch.cuda.synchronize()
        
        print(f"Running {NUM_STEPS} steps...")
        start_time = time.time()
        for _ in range(NUM_STEPS):
            trainer.train_step(input_ids, labels=input_ids)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / NUM_STEPS
        print(f"LEMA Avg Time/Step: {avg_time:.4f}s")
        
        return avg_time
        
    except Exception as e:
        print(f"LEMA Benchmark Failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')
    finally:
        if os.path.exists(model_info['path']):
            os.remove(model_info['path'])

if __name__ == "__main__":
    results = {}
    for model in MODELS:
        peft_time = benchmark_peft_speed(model)
        lema_time = benchmark_lema_speed(model)
        results[model['name']] = {"PEFT": peft_time, "LEMA": lema_time}
    
    print("\n=== SPEED BENCHMARK RESULTS (Time per Step) ===")
    print(f"{ 'Model':<20} | { 'PEFT (s)':<10} | { 'LEMA (s)':<10} | { 'Overhead':<10}")
    print("-" * 60)
    for name, data in results.items():
        peft = data["PEFT"]
        lema = data["LEMA"]
        overhead = (lema / peft) if peft > 0 else float('inf')
        print(f"{name:<20} | {peft:<10.4f} | {lema:<10.4f} | {overhead:<10.2f}x")

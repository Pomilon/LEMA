import torch
import gc
import os
import time
import transformers
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig
from safetensors.torch import save_file

# LEMA Unified Imports
from lema import LemaConfig, LemaModel, MemoryStrategy
from lema.utils.model_utils import prepare_monolithic_safetensors

print(f"Using Transformers version: {transformers.__version__}")

MODELS = [
    {"name": "TinyLlama 1.1B", "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "path": "tinyllama_1b.safetensors", "type": "llama"},
    {"name": "Llama-2 7B", "hf_id": "NousResearch/Llama-2-7b-hf", "path": "llama2_7b.safetensors", "type": "llama"}
]

NUM_STEPS = 20 

def benchmark_peft_speed(model_info):
    print(f"\n>>> BENCHMARKING PEFT SPEED: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_info['hf_id'], torch_dtype=torch.float16, device_map="cuda")
        model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            r=16, lora_alpha=32, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, 1000, (1, 512)).cuda()
        
        # Warmup
        model(input_ids, labels=input_ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(NUM_STEPS):
            model(input_ids, labels=input_ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / NUM_STEPS
        print(f"PEFT Avg Time/Step: {avg_time:.4f}s")
        return avg_time
    except Exception as e:
        print(f"PEFT Benchmark Failed: {e}")
        return float('inf')

def benchmark_lema_speed(model_info):
    print(f"\n>>> BENCHMARKING LEMA SPEED: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    
    if not os.path.exists(model_info['path']):
        print(f"Preparing {model_info['path']}...")
        prepare_monolithic_safetensors(model_info['hf_id'], model_info['path'])
    
    use_gc = "7b" in model_info['hf_id'].lower()
    
    try:
        config = LemaConfig(
            model_name_or_path=model_info['hf_id'],
            gbi_path=model_info['path'],
            device="cuda",
            strategy=MemoryStrategy.STREAMING,
            learning_rate=1e-4,
            gradient_checkpointing=use_gc,
            lora_rank=16
        )
        
        model = LemaModel(config)
        model.initialize_lora()
        
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=config.learning_rate)
        trainer = model.get_trainer(optimizer)
        
        input_ids = torch.randint(0, 1000, (1, 512)).cuda()
        trainer.train_step(input_ids, labels=input_ids) # Warmup
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(NUM_STEPS):
            trainer.train_step(input_ids, labels=input_ids)
        torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / NUM_STEPS
        print(f"LEMA Avg Time/Step: {avg_time:.4f}s")
        return avg_time
    except Exception as e:
        print(f"LEMA Benchmark Failed: {e}")
        return float('inf')
    finally:
        if os.path.exists(model_info['path']): os.remove(model_info['path'])

if __name__ == "__main__":
    results = {}
    for model in MODELS:
        peft = benchmark_peft_speed(model)
        lema = benchmark_lema_speed(model)
        results[model['name']] = {"PEFT": peft, "LEMA": lema}
    
    print("\n=== UNIFIED SPEED BENCHMARK RESULTS ===")
    for name, data in results.items():
        p, l = data["PEFT"], data["LEMA"]
        overhead = (l / p) if p > 0 and p != float('inf') else float('inf')
        print(f"{name}: PEFT={p:.4f}s, LEMA={l:.4f}s, Overhead={overhead:.2f}x")

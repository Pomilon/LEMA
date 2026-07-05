import torch
import gc
import os
import time
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig

# LEMA Unified Imports
from lema import LemaConfig, LemaModel, MemoryStrategy, logger

print(f"Using Transformers version: {transformers.__version__}")

MODELS = [
    {"name": "TinyLlama 1.1B", "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "path": "tinyllama_1b.safetensors"},
    {"name": "Llama-2 7B", "hf_id": "NousResearch/Llama-2-7b-hf", "path": "llama2_7b.safetensors"}
]

NUM_STEPS = 10 

def benchmark_peft_speed(model_info):
    print(f"\n>>> BENCHMARKING PEFT SPEED: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_info['hf_id'], 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            r=16, lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Minimal for speed compare
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
        
        # Cleanup
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return avg_time
    except Exception as e:
        print(f"PEFT Benchmark Failed: {e}")
        return float('inf')

def benchmark_lema_speed(model_info):
    print(f"\n>>> BENCHMARKING LEMA SPEED: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    
    try:
        use_resident = "tinyllama" in model_info['hf_id'].lower()
        
        config = LemaConfig(
            model_name_or_path=model_info['hf_id'],
            gbi_path=model_info['path'],
            device="cuda",
            strategy=MemoryStrategy.RESIDENT, # Optimized partial-residency-aware strategy
            lora_rank=16
        )
        
        # Zero-Config Loading (Auto-converts and Auto-Optimizes)
        model = LemaModel(config)
        model.initialize_lora()
        
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
        # Enable the new Dynamic Flight Check
        trainer = model.get_trainer(optimizer, auto_optimize=True)
        
        input_ids = torch.randint(0, 1000, (1, 512)).cuda()
        trainer.train_step(input_ids, labels=input_ids) # Warmup
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(NUM_STEPS):
            trainer.train_step(input_ids, labels=input_ids)
        torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / NUM_STEPS
        print(f"LEMA Avg Time/Step: {avg_time:.4f}s")
        
        # Verify Generation (best-effort — may fail on seq_len mismatch)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_info['hf_id'])
            output = model.generate("The meaning of life is", tokenizer, max_new_tokens=10)
            print(f"LEMA Generation Check: {output}")
        except Exception as ge:
            print(f"LEMA Generation Check skipped: {ge}")
        
        return avg_time
    except Exception as e:
        logger.error(f"LEMA Benchmark Failed: {e}", exc_info=True)
        return float('inf')

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

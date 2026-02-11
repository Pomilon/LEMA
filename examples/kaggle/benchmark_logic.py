import torch
import gc
import os
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors.torch import save_file

# LEMA Imports
from src.lema.core.gbi import GlobalBinaryIndex
from src.lema.models.llama import LlamaAdapter
from src.lema.models.gpt2 import GPT2Adapter
from src.lema.engine.trainer import LemaTrainer
from src.lema.core.lora import LoRAManager
from src.lema.config import LemaConfig, MemoryStrategy

print(f"Using Transformers version: {transformers.__version__}")

# --- MODELS TO TEST ---
MODELS = [
    {
        "name": "GPT2 (Small)",
        "hf_id": "gpt2",
        "path": "gpt2.safetensors",
        "type": "gpt2"
    },
    {
        "name": "TinyLlama 1.1B",
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "path": "tinyllama_1b.safetensors",
        "type": "llama"
    },
    {
        "name": "SmolLM2 1.7B",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B",
        "path": "smollm2_1.7b.safetensors",
        "type": "llama"
    },
    {
        "name": "Llama-2 7B",
        "hf_id": "NousResearch/Llama-2-7b-hf",
        "path": "llama2_7b.safetensors",
        "type": "llama"
    }
]

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
    elif hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        # GPT2 shared weights
        if model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr():
             model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

    print(f"Saving to {model_info['path']}...")
    save_file(model.state_dict(), model_info['path'])
    del model
    gc.collect()

from peft import get_peft_model, LoraConfig

def run_peft_baseline(model_info):
    print(f"\n>>> TESTING STANDARD PEFT ON: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    download_and_convert(model_info)
    
    try:
        # Load Model in FP16
        model = AutoModelForCausalLM.from_pretrained(
            model_info['hf_id'],
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        
        # Configure LoRA
        target_modules = ["c_attn", "c_proj", "c_fc"] if model_info['type'] == "gpt2" else \
                         ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                         
        peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, peft_config)
        print(f"PEFT Trainable params: {model.print_trainable_parameters()}")
        
        # Dummy Train Step
        input_ids = torch.randint(0, 1000, (1, 128)).cuda()
        output = model(input_ids, labels=input_ids)
        loss = output.loss
        loss.backward()
        
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Standard PEFT Peak VRAM: {peak_vram:.2f} GB")
        
        del model
        torch.cuda.empty_cache()
        return peak_vram
        
    except Exception as e:
        print(f"PEFT Baseline Failed: {e}")
        return float('inf')

def run_test(model_info):
    print(f"\n>>> TESTING LEMA ON: {model_info['name']} <<<")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    download_and_convert(model_info)
    
    try:
        # 1. Config
        hf_config = AutoConfig.from_pretrained(model_info['hf_id'])
        hf_config_dict = hf_config.to_dict()
        hf_config_dict["attn_implementation"] = "eager"
        hf_config_dict["torch_dtype"] = "float16"
        
        # 2. Components
        if model_info['type'] == "llama":
            adapter = LlamaAdapter(hf_config_dict)
        elif model_info['type'] == "gpt2":
            adapter = GPT2Adapter(hf_config_dict)
        else:
            raise ValueError(f"Unknown type: {model_info['type']}")
            
        gbi = GlobalBinaryIndex(model_info['path'])
        
        # LEMA Config
        lema_config = LemaConfig(
            model_name_or_path=model_info['path'],
            device="cuda",
            strategy=MemoryStrategy.STREAMING,
            learning_rate=1e-4
        )
        
        # LoRA - ALL LINEAR LAYERS
        target_modules = ["c_attn", "c_proj", "c_fc"] if model_info['type'] == "gpt2" else \
                         ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                         
        lora_config = {
            "r": 16, "alpha": 32, 
            "target_modules": target_modules
        }
        lora_manager = LoRAManager(lora_config, device="cuda")
        
        # Initialize LoRA params (Fix leak)
        print("Initializing LoRA parameters...")
        for layer in adapter.get_layer_metadata():
            if layer['type'] == 'block':
                module = adapter.construct_layer_module(layer['id'], None, lora_manager)
                adapter.release_layer_module(module)
        
        torch.cuda.empty_cache()
        
        trainable_params = lora_manager.get_trainable_parameters()
        print(f"Trainable Tensors: {len(trainable_params)}")
        optimizer = torch.optim.AdamW(trainable_params, lr=lema_config.learning_rate)
        
        trainer = LemaTrainer(
            config=lema_config,
            model_adapter=adapter, 
            gbi=gbi, 
            lora_manager=lora_manager, 
            optimizer=optimizer
        )
        
        # 3. Execution
        print("Executing Train Step...")
        # Create dummy inputs based on vocab size
        vocab_size = hf_config.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 128)).cuda()
        
        logits, loss = trainer.train_step(input_ids, labels=input_ids)
        
        print(f"Loss: {loss:.4f}")
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"LEMA Peak VRAM: {peak_vram:.2f} GB")
        print(f"RESULT: {model_info['name']} -> SUCCESS")
        return peak_vram
        
    except Exception as e:
        print(f"RESULT: {model_info['name']} -> FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')
    finally:
        # Cleanup model file to save disk space on Kaggle
        if os.path.exists(model_info['path']):
            os.remove(model_info['path'])
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    results = {}
    for model in MODELS:
        peft_vram = run_peft_baseline(model)
        lema_vram = run_test(model)
        results[model['name']] = {"PEFT": peft_vram, "LEMA": lema_vram}
    
    print("\n=== FINAL RESULTS (VRAM in GB) ===")
    print(f"{'Model':<20} | {'PEFT (Baseline)':<15} | {'LEMA (Ours)':<15} | {'Savings':<10}")
    print("-" * 65)
    for name, data in results.items():
        peft = data["PEFT"]
        lema = data["LEMA"]
        savings = (1 - lema/peft) * 100 if peft > 0 else 0
        print(f"{name:<20} | {peft:<15.2f} | {lema:<15.2f} | {savings:<10.1f}%")
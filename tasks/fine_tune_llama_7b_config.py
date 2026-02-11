import torch
import os
import time
from transformers import AutoTokenizer, AutoConfig
from src.lema.core.gbi import GlobalBinaryIndex
from src.lema.models.llama import LlamaAdapter
from src.lema.engine.trainer import LemaTrainer
from src.lema.core.lora import LoRAManager
from src.lema.config import LemaConfig, MemoryStrategy

MODEL_NAME = "NousResearch/Llama-2-7b-hf"
MODEL_PATH = "llama2_7b.safetensors"

TRAINING_DATA = [
    "What is photosynthesis? Photosynthesis is the process by which plants use sunlight to synthesize nutrients from carbon dioxide and water.",
    "Who was Albert Einstein? Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "What is the capital of France? The capital of France is Paris.",
    "Explain gravity. Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
    "What is LEMA? LEMA is a framework that virtualizes GPU memory to enable training large models on limited hardware.",
] * 10

def fine_tune_llama_7b_with_config():
    print("--- STARTING LEMA 7B FINE-TUNING (Config Object) ---")
    
    # 1. Setup Configuration
    config = LemaConfig(
        model_name_or_path=MODEL_PATH,
        device="cuda",
        strategy=MemoryStrategy.STREAMING,
        lora_rank=16,
        lora_alpha=32,
        learning_rate=5e-5,
        dtype="float16"
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # HF Config (for model architecture)
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config = hf_config.to_dict()
    model_config["attn_implementation"] = config.attn_implementation
    model_config["torch_dtype"] = config.dtype
    
    # Components
    adapter = LlamaAdapter(model_config)
    gbi = GlobalBinaryIndex(config.gbi_path)
    
    # LoRA
    lora_config_dict = {
        "r": config.lora_rank, 
        "alpha": config.lora_alpha, 
        "target_modules": config.lora_target_modules
    }
    lora_manager = LoRAManager(lora_config_dict, device=config.device)
    
    print("Initializing LoRA parameters...")
    for layer in adapter.get_layer_metadata():
        if layer['type'] == 'block':
            module = adapter.construct_layer_module(layer['id'], None, lora_manager)
            adapter.release_layer_module(module)
    torch.cuda.empty_cache()
    
    trainable_params = lora_manager.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    
    # Trainer
    trainer = LemaTrainer(
        config=config,
        model_adapter=adapter, 
        gbi=gbi, 
        lora_manager=lora_manager, 
        optimizer=optimizer
    )
    
    # Training Loop
    print(f"\nTraining on {len(TRAINING_DATA)} examples...")
    for epoch in range(1):
        total_loss = 0
        for i, text in enumerate(TRAINING_DATA):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(config.device)
            
            logits, loss = trainer.train_step(input_ids, labels=input_ids)
            total_loss += loss
            
            if (i+1) % 10 == 0:
                print(f"Step {i+1}/{len(TRAINING_DATA)} - Current Loss: {loss:.4f}")
                
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(TRAINING_DATA):.4f}")
    
    print("Fine-tuning with Config Object completed successfully.")

    # Cleanup
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

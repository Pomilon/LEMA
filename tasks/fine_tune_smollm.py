import torch
import os
import time
from transformers import AutoTokenizer, AutoConfig
from src.lema.core.gbi import GlobalBinaryIndex
from src.lema.models.llama import LlamaAdapter
from src.lema.engine.trainer import LemaTrainer
from src.lema.core.lora import LoRAManager
from src.lema.config import LemaConfig, MemoryStrategy

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_PATH = "smollm2_1.7b.safetensors"

# 1. Realistic Dataset: "Concise Assistant"
# The model should learn to answer everything in one short, professional sentence.
TRAINING_DATA = [
    "What is photosynthesis? Photosynthesis is the process by which plants use sunlight to synthesize nutrients from carbon dioxide and water.",
    "Who was Albert Einstein? Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "What is the capital of France? The capital of France is Paris.",
    "Explain gravity. Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
    "What is LEMA? LEMA is a framework that virtualizes GPU memory to enable training large models on limited hardware.",
    "How does a CPU work? A CPU executes instructions of a computer program by performing basic arithmetic, logic, and I/O operations.",
    "What is the speed of light? The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Define machine learning. Machine learning is a field of artificial intelligence focused on building systems that learn from data.",
    "What is DNA? DNA is a molecule that carries the genetic instructions used in the growth, development, and functioning of all living organisms.",
    "What is the ocean? The ocean is a continuous body of salt water that covers more than 70 percent of Earth's surface.",
] * 10 # 100 examples for more "realistic" weight updates

def fine_tune_realistic():
    print("--- STARTING REALISTIC LEMA FINE-TUNING (v0.6) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    hf_config_dict = config.to_dict()
    hf_config_dict["attn_implementation"] = "eager"
    
    adapter = LlamaAdapter(hf_config_dict)
    gbi = GlobalBinaryIndex(MODEL_PATH)
    
    # LEMA Config
    lema_config = LemaConfig(
        model_name_or_path=MODEL_PATH,
        device="cuda",
        strategy=MemoryStrategy.STREAMING
    )
    
    # 2. HEAVY LoRA Config (All major linear layers)
    # This increases the number of parameters the optimizer has to manage.
    lora_config = {
        "r": 16, 
        "alpha": 32, 
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
    lora_manager = LoRAManager(lora_config, device="cuda")
    
    print("Initializing heavy LoRA parameters...")
    # Trigger param creation for all layers
    for layer in adapter.get_layer_metadata():
        if layer['type'] == 'block':
            module = adapter.construct_layer_module(layer['id'], None, lora_manager)
            adapter.release_layer_module(module)
    
    # Clear cache after initialization
    torch.cuda.empty_cache()
            
    trainable_params = lora_manager.get_trainable_parameters()
    print(f"Number of trainable LoRA parameters: {len(trainable_params)}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    
    trainer = LemaTrainer(
        config=lema_config,
        model_adapter=adapter, 
        gbi=gbi, 
        lora_manager=lora_manager, 
        optimizer=optimizer
    )
    
    # 3. Training Loop
    print("\nTraining on 100 examples...")
    start_time = time.time()
    for epoch in range(3): # Fewer epochs but more data per epoch
        total_loss = 0
        for i, text in enumerate(TRAINING_DATA):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to("cuda")
            
            logits, loss = trainer.train_step(input_ids, labels=input_ids)
            total_loss += loss
            
            if (i+1) % 20 == 0:
                print(f"Step {i+1}/{len(TRAINING_DATA)} - Current Loss: {loss:.4f}")
                
        avg_loss = total_loss / len(TRAINING_DATA)
        print(f"Epoch {epoch+1}/3 - Avg Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")
        
    # 4. Validation
    print("\n--- TESTING BEHAVIOR (Concise Assistant) ---")
    test_prompts = [
        "What is the moon?",
        "Who was Isaac Newton?"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        
        generated = input_ids
        for _ in range(25):
            with torch.no_grad():
                logits, _ = trainer.train_step(generated)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token_id], dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
        print(f"Prompt: {prompt}")
        print(f"Response: {tokenizer.decode(generated[0], skip_special_tokens=True)}")

    # Final Cleanup
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

if __name__ == "__main__":
    # Expecting benchmark script to have downloaded the model already
    if os.path.exists(MODEL_PATH):
        fine_tune_realistic()
    else:
        print(f"Error: {MODEL_PATH} not found.")
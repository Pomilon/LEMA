import torch
import os
import time
from transformers import AutoTokenizer
from lema import LemaConfig, LemaModel, MemoryStrategy
from lema.utils.model_utils import prepare_monolithic_safetensors

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_PATH = "smollm2_1.7b.safetensors"

TRAINING_DATA = [
    "What is photosynthesis? Photosynthesis is the process by which plants use sunlight to synthesize nutrients from carbon dioxide and water.",
    "Who was Albert Einstein? Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "What is the capital of France? The capital of France is Paris.",
] * 10

def fine_tune_smollm():
    print("--- STARTING SMOL-LM FINE-TUNING ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Preparing {MODEL_PATH}...")
        prepare_monolithic_safetensors(MODEL_NAME, MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = LemaConfig(
        model_name_or_path=MODEL_NAME,
        gbi_path=MODEL_PATH,
        device="cuda",
        strategy=MemoryStrategy.STREAMING,
        learning_rate=5e-5,
        lora_rank=16
    )
    
    model = LemaModel(config)
    model.initialize_lora()
    
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=config.learning_rate)
    trainer = model.get_trainer(optimizer)
    
    start_time = time.time()
    for text in TRAINING_DATA:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to("cuda")
        trainer.train_step(input_ids, labels=input_ids)
            
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    trainer.save_checkpoint("output/smollm-final")

if __name__ == "__main__":
    fine_tune_smollm()

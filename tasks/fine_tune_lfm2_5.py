"""
LEMA Fine-Tuning: LFM2.5 8B A1B

Fine-tunes LFM2.5 8B A1B on a T4 16GB GPU using LEMA's sliding-window streaming.
Demonstrates training an 8B MoE model on low-VRAM hardware.
"""
import torch
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from transformers import AutoTokenizer
from lema import LemaConfig, LemaModel, MemoryStrategy, logger

def fine_tune_lfm2_5():
    logger.info("--- STARTING LEMA LFM2.5 8B A1B FINE-TUNING ---")

    hf_id = "LiquidAI/LFM2.5-8B-A1B"
    model_path = "lfm2_5_8b.safetensors"

    config = LemaConfig(
        model_name_or_path=hf_id,
        gbi_path=model_path,
        device="cuda",
        strategy=MemoryStrategy.STREAMING,
        lora_rank=8,
        gradient_checkpointing=True,
        prefetch_distance=2,
    )

    model = LemaModel(config)
    model.initialize_lora()

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
    trainer = model.get_trainer(optimizer)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    texts = [
        "The meaning of life is",
        "Machine learning is a field of",
        "The capital of France is",
        "In the beginning, there was",
        "The theory of relativity states that",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")

    num_epochs = 3
    for epoch in range(num_epochs):
        for i in range(0, inputs["input_ids"].size(0), 1):
            input_ids = inputs["input_ids"][i:i+1]
            labels = input_ids.clone()

            t0 = time.perf_counter()
            logits, loss = trainer.train_step(input_ids, labels=labels)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000

            logger.info(f"Epoch {epoch+1}/{num_epochs} | Step {i+1} | Loss: {loss:.4f} | Time: {dt:.0f}ms")

    logger.info("LFM2.5 fine-tuning complete!")
    return model

if __name__ == "__main__":
    model = fine_tune_lfm2_5()

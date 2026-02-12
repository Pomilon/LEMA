import torch
import os
from lema import LemaConfig, LemaModel, MemoryStrategy
from lema.utils.model_utils import break_shared_weights
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

def run_demo():
    print("--- LEMA Unified API Demo ---")

    model_dir = "./demo_model"
    gbi_path = os.path.join(model_dir, "model.safetensors")
    
    # 1. Configuration
    config = LemaConfig(
        model_name_or_path=model_dir, 
        model_type="gpt2",
        gbi_path=gbi_path, 
        device="cpu",
        strategy=MemoryStrategy.STREAMING,
        lora_rank=8,
        lora_target_modules=["c_attn"],
        output_dir="./lema_checkpoints",
        save_steps=10
    )

    # 2. Initialize Model & Trainer
    model = LemaModel(config)
    model.initialize_lora()
    
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
    trainer = model.get_trainer(optimizer)

    # 3. Execution
    print("Executing training step...")
    input_ids = torch.randint(0, 1000, (1, 16))
    logits, loss = trainer.train_step(input_ids, labels=input_ids)
    
    print(f"Step complete. Loss: {loss:.4f}")
    trainer.save_checkpoint("./lema_checkpoints/final_demo")

if __name__ == "__main__":
    model_dir = "./demo_model"
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
        print("Generating dummy model...")
        os.makedirs(model_dir, exist_ok=True)
        dummy_config = GPT2Config(n_layer=2, n_embd=128, n_head=4, vocab_size=1000)
        dummy_model = GPT2LMHeadModel(dummy_config)
        dummy_model = break_shared_weights(dummy_model)
        
        state_dict = {k: v.clone().detach() for k, v in dummy_model.state_dict().items()}
        save_file(state_dict, os.path.join(model_dir, "model.safetensors"))
        dummy_config.save_pretrained(model_dir)
    
    run_demo()
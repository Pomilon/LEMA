import torch
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
from lema import LemaConfig, LemaModel, MemoryStrategy
from lema.utils.model_utils import break_shared_weights
import os
import pytest

def test_lora_training_loop(tmp_path):
    # 1. Setup Model Directory
    model_dir = tmp_path / "gpt2_model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"
    
    config_hf = GPT2Config(
        vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=2,
        attn_implementation="eager"
    )
    model_hf = GPT2LMHeadModel(config_hf)
    model_hf = break_shared_weights(model_hf)
    
    state_dict = {k: v.clone().detach() for k, v in model_hf.state_dict().items()}
    save_file(state_dict, str(model_path))
    config_hf.save_pretrained(str(model_dir))
    
    # 2. Unified LEMA API
    lema_config = LemaConfig(
        model_name_or_path=str(model_dir),
        model_type="gpt2",
        gbi_path=str(model_path),
        device="cpu",
        strategy=MemoryStrategy.STREAMING,
        learning_rate=0.1,
        lora_rank=2,
        lora_target_modules=["c_attn"],
        save_steps=2,
        output_dir=str(tmp_path / "checkpoints")
    )
    
    model = LemaModel(lema_config)
    model.initialize_lora()
    
    optimizer = optim.SGD(model.get_trainable_parameters(), lr=lema_config.learning_rate)
    trainer = model.get_trainer(optimizer)
    
    # 3. Training
    input_ids = torch.randint(0, 100, (1, 10))
    initial_params = [p.clone().detach() for p in model.get_trainable_parameters()]
    
    for _ in range(3):
        trainer.train_step(input_ids, labels=input_ids)
        
    # 4. Verification
    params_changed = False
    for p_init, p_curr in zip(initial_params, model.get_trainable_parameters()):
        if not torch.allclose(p_init, p_curr):
            params_changed = True
            break
            
    assert params_changed, "LoRA parameters did not change!"
    assert (tmp_path / "checkpoints" / "checkpoint-2").exists()

if __name__ == "__main__":
    import sys
    # Local run support
    pytest.main([__file__])

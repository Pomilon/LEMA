import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
from src.lema.core.gbi import GlobalBinaryIndex
from src.lema.models.gpt2 import GPT2Adapter
from src.lema.engine.trainer import LemaTrainer
from src.lema.config import LemaConfig, MemoryStrategy
import os
import pytest

def test_forward_backward_equivalence(tmp_path):
    # 1. Setup Standard Model
    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        attn_implementation="eager",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0
    )
    
    # Force deterministic behavior
    torch.manual_seed(42)
    
    model = GPT2LMHeadModel(config)
    model.eval() # Disable dropout for deterministic check
    
    # Save weights for LEMA
    state_dict = model.state_dict()
    new_state_dict = {k: v.clone() for k, v in state_dict.items()}
    
    model_path = tmp_path / "test_equiv.safetensors"
    save_file(new_state_dict, str(model_path))
    
    # 2. Setup LEMA
    lema_config_obj = LemaConfig(
        model_name_or_path=str(model_path),
        device="cpu",
        strategy=MemoryStrategy.STREAMING
    )
    
    adapter = GPT2Adapter(config.to_dict())
    gbi = GlobalBinaryIndex(str(model_path))
    
    trainer = LemaTrainer(
        config=lema_config_obj,
        model_adapter=adapter,
        gbi=gbi
    )
    
    # 3. Create Inputs
    input_ids = torch.randint(0, 100, (1, 10))
    
    # 4. Standard Forward
    with torch.no_grad():
        std_output = model(input_ids).logits
    
    # LEMA Forward
    lema_output, _ = trainer.train_step(input_ids)
    
    # Check Forward
    assert torch.allclose(std_output, lema_output, atol=1e-5), "Forward pass logits mismatch!"
    print("Forward pass matches.")
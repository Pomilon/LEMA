import torch
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
from src.lema.core.gbi import GlobalBinaryIndex
from src.lema.models.gpt2 import GPT2Adapter
from src.lema.engine.trainer import LemaTrainer
from src.lema.core.lora import LoRAManager
from src.lema.config import LemaConfig, MemoryStrategy
import os
import pytest

def test_lora_training_loop(tmp_path):
    # 1. Setup
    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        attn_implementation="eager"
    )
    
    # Save dummy weights
    model = GPT2LMHeadModel(config)
    state_dict = model.state_dict()
    new_state_dict = {k: v.clone() for k, v in state_dict.items()}
    model_path = tmp_path / "test_train.safetensors"
    save_file(new_state_dict, str(model_path))
    
    # 2. Components
    lema_config = LemaConfig(
        model_name_or_path=str(model_path),
        device="cpu",
        strategy=MemoryStrategy.STREAMING,
        learning_rate=0.1
    )
    
    adapter = GPT2Adapter(config.to_dict())
    gbi = GlobalBinaryIndex(str(model_path))
    
    lora_config = {"r": 2, "alpha": 4, "target_modules": ["c_attn"]}
    lora_manager = LoRAManager(lora_config, device="cpu")
    
    # Trigger param creation
    for layer in adapter.get_layer_metadata():
        if layer['type'] == 'block':
            module = adapter.construct_layer_module(layer['id'], None, lora_manager)
            adapter.release_layer_module(module)

    params = lora_manager.get_trainable_parameters()
    optimizer = optim.SGD(params, lr=lema_config.learning_rate)
    
    trainer = LemaTrainer(
        config=lema_config,
        model_adapter=adapter,
        gbi=gbi,
        lora_manager=lora_manager,
        optimizer=optimizer
    )
    
    # Capture initial param values
    initial_params = [p.clone().detach() for p in params]
    
    # 3. Train Loop
    input_ids = torch.randint(0, 100, (1, 10))
    losses = []
    for _ in range(3):
        # We need to provide labels to get a real loss if we want, 
        # or it will use dummy .mean()
        _, loss = trainer.train_step(input_ids)
        losses.append(loss if loss is not None else 0.0)
        
    # 4. Verify Updates
    params_changed = False
    for p_init, p_curr in zip(initial_params, params):
        if not torch.allclose(p_init, p_curr):
            params_changed = True
            break
            
    assert params_changed, "LoRA parameters did not change after training!"
    print(f"Losses: {losses}")

if __name__ == "__main__":
    test_lora_training_loop(os.path.abspath("."))
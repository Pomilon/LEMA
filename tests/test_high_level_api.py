import torch
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
from lema.core.model import LemaModel
from lema.engine.trainer import LemaTrainer
from lema.config import LemaConfig, MemoryStrategy
import os
import shutil

def test_high_level_api(tmp_path):
    # 1. Setup Dummy Model
    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        attn_implementation="eager"
    )
    
    # Save dummy weights
    model_hf = GPT2LMHeadModel(config)
    state_dict = model_hf.state_dict()
    # Clone to avoid shared memory issues in safetensors
    safe_state_dict = {k: v.clone().detach() for k, v in state_dict.items()}
    
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(safe_state_dict, str(model_path))
    config.save_pretrained(str(model_dir))
    
    # 2. Initialize LemaModel
    lema_config = LemaConfig(
        model_name_or_path=str(model_dir),
        model_type="gpt2",
        gbi_path=str(model_path),
        device="cpu",
        strategy=MemoryStrategy.STREAMING,
        learning_rate=0.1,
        lora_rank=2,
        lora_target_modules=["c_attn"]
    )
    
    lema_model = LemaModel(lema_config)
    lema_model.initialize_lora()
    
    # 3. Setup Trainer
    optimizer = optim.SGD(lema_model.get_trainable_parameters(), lr=lema_config.learning_rate)
    trainer = LemaTrainer(
        config=lema_model.config,
        model_adapter=lema_model.adapter,
        gbi=lema_model.gbi,
        lora_manager=lema_model.lora_manager,
        optimizer=optimizer
    )
    
    # 4. Train Step
    input_ids = torch.randint(0, 100, (1, 10))
    labels = input_ids.clone()
    _, loss1 = trainer.train_step(input_ids, labels=labels)
    
    initial_params = [p.clone().detach() for p in lema_model.get_trainable_parameters()]
    
    for _ in range(2):
        trainer.train_step(input_ids, labels=labels)
        
    final_params = [p.clone().detach() for p in lema_model.get_trainable_parameters()]
    
    # Verify updates
    params_changed = False
    for p_init, p_curr in zip(initial_params, final_params):
        if not torch.allclose(p_init, p_curr):
            params_changed = True
            break
    assert params_changed, "Parameters did not change"
    
    # 5. Save and Load
    save_dir = tmp_path / "saved_model"
    lema_model.save_pretrained(str(save_dir))
    
    assert os.path.exists(save_dir / "lema_config.json")
    assert os.path.exists(save_dir / "adapter_model.bin")
    
    # Load back
    loaded_model = LemaModel.from_pretrained(str(save_dir))
    
    # Verify loaded params match final_params
    loaded_params = loaded_model.get_trainable_parameters()
    for p_final, p_loaded in zip(final_params, loaded_params):
        assert torch.allclose(p_final, p_loaded)

if __name__ == "__main__":
    from types import SimpleNamespace
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_high_level_api(SimpleNamespace(
            __truediv__=lambda self, other: os.path.join(self.path, str(other)),
            path=tmp
        ))
    print("Test Passed!")

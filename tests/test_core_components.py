import pytest
import torch
import torch.nn as nn
from src.lema.core.gbi import GlobalBinaryIndex
from src.lema.core.lora import LoRAManager, LoRAWrapper
from src.lema.core.memory import TripleBufferManager

# Mocking
class MockAdapter:
    def get_param_names_for_layer(self, layer_id):
        return ["test.weight"]

def test_gbi_loading(tmp_path):
    # Create a dummy safetensors file
    from safetensors.torch import save_file
    d = tmp_path / "test_model.safetensors"
    t = torch.randn(10, 10)
    save_file({"test.weight": t}, str(d))
    
    gbi = GlobalBinaryIndex(str(d))
    loaded = gbi.load_tensors(["test.weight"])
    assert "test.weight" in loaded
    assert torch.equal(loaded["test.weight"], t)

def test_lora_injection():
    # Setup a simple model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    config = {
        "target_modules": ["0", "2"], # targeting the Linear layers by name index? 
        # Named children of Sequential are "0", "1", "2".
        "r": 4,
        "alpha": 8
    }
    
    # Update config to target standard Linear names if we were using names
    # But for Sequential, names are indices.
    
    manager = LoRAManager(config, device="cpu")
    manager.apply_lora(0, model)
    
    # Check if replaced
    assert isinstance(model[0], LoRAWrapper)
    assert isinstance(model[2], LoRAWrapper)
    assert not isinstance(model[1], LoRAWrapper)
    
    # Check shapes
    # Linear(10, 10) -> A: [4, 10], B: [10, 4] (transposed logic in LoRAWrapper)
    # LoRAWrapper expects lora_A: [rank, in], lora_B: [out, rank]
    assert model[0].lora_A.shape == (4, 10)
    assert model[0].lora_B.shape == (10, 4)

import torch
import os
from typing import Optional, Dict, Any, Union
from transformers import AutoConfig

from ..config import LemaConfig
from ..models import get_adapter
from .gbi import GlobalBinaryIndex
from .lora import LoRAManager
from .memory import TripleBufferManager

class LemaModel:
    """
    High-level interface for LEMA Models.
    Wraps all low-level components into a single object.
    """
    def __init__(self, config: LemaConfig):
        self.config = config
        
        # 1. Initialize GBI
        self.gbi = GlobalBinaryIndex(config.gbi_path)
        
        # 2. Get HF config for the adapter
        # Try to load from model path or fallback to a default if not found
        try:
            hf_config_obj = AutoConfig.from_pretrained(config.model_name_or_path)
            hf_config_dict = hf_config_obj.to_dict()
        except:
            # Fallback to config dict if AutoConfig fails
            hf_config_dict = config.to_dict()

        # 3. Initialize Adapter
        model_type = config.model_type
        if model_type is None:
            # Auto-detect from path
            path_lower = config.model_name_or_path.lower()
            if "llama" in path_lower or "smollm" in path_lower:
                model_type = "llama"
            elif "gpt2" in path_lower:
                model_type = "gpt2"
            else:
                # Default to llama if unknown but looks like it
                model_type = "llama"
        
        self.adapter = get_adapter(model_type, hf_config_dict)
        
        # 4. Initialize LoRA Manager
        self.lora_manager = LoRAManager({
            "r": config.lora_rank,
            "alpha": config.lora_alpha,
            "target_modules": config.lora_target_modules
        }, device=config.device)
        
        # 5. Initialize Memory Manager
        self.memory = TripleBufferManager(
            self.gbi, 
            self.adapter, 
            device=config.device, 
            strategy=config.strategy
        )

    def get_trainer(self, optimizer: torch.optim.Optimizer):
        """Returns a LemaTrainer instance pre-configured with this model's components."""
        from ..engine.trainer import LemaTrainer
        return LemaTrainer(
            config=self.config,
            model_adapter=self.adapter,
            gbi=self.gbi,
            lora_manager=self.lora_manager,
            optimizer=optimizer,
            memory_manager=self.memory
        )

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Loads a LEMA model and its adapters from a directory."""
        config = LemaConfig.from_pretrained(path, **kwargs)
        model = cls(config)
        
        # Load adapters if they exist
        if os.path.exists(os.path.join(path, "adapter_model.bin")):
            model.lora_manager.load_pretrained(path)
            
        return model

    def save_pretrained(self, save_directory: str):
        """Saves the configuration and LoRA adapters."""
        self.config.save_pretrained(save_directory)
        self.lora_manager.save_pretrained(save_directory)

    def initialize_lora(self):
        """Pre-initializes all LoRA adapters by constructing and releasing each layer once."""
        for layer in self.adapter.get_layer_metadata():
            if layer['type'] == 'block':
                module = self.adapter.construct_layer_module(layer['id'], None, self.lora_manager)
                if hasattr(self.adapter, "release_layer_module"):
                    self.adapter.release_layer_module(module)

    def get_trainable_parameters(self):
        return self.lora_manager.get_trainable_parameters()

    def to(self, device: str):
        self.config.device = device
        self.lora_manager.device = device
        self.memory.device = device
        return self
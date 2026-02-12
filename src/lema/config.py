from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class MemoryStrategy(Enum):
    STREAMING = "streaming" # Disk -> RAM -> VRAM
    RESIDENT = "resident"   # RAM -> VRAM (No Disk offload for weights)

@dataclass
class LemaConfig:
    """
    Central Configuration for LEMA Training/Inference.
    """
    # Model Settings
    model_name_or_path: str
    model_type: Optional[str] = None # 'llama' or 'gpt2', auto-detected if None
    gbi_path: Optional[str] = None # Path to converted safetensors for GBI
    
    # Hardware / Memory Settings
    device: str = "cuda"
    strategy: MemoryStrategy = MemoryStrategy.STREAMING
    ram_buffer_size: int = 2 # Number of layers to keep in RAM
    vram_buffer_size: int = 1 # Number of layers to keep in VRAM
    
    # LoRA Settings
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Training Settings
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    gradient_checkpointing: bool = False
    
    # Checkpointing Settings
    save_steps: int = 500
    output_dir: str = "output"
    
    # Advanced
    dtype: str = "float16" # float16, bfloat16, float32
    attn_implementation: str = "eager" # eager, sdpa, flash_attention_2

    def __post_init__(self):
        if self.gbi_path is None:
            # Default to expecting a local safetensors file named after the model or a standard name
            if self.model_name_or_path.endswith(".safetensors"):
                self.gbi_path = self.model_name_or_path
            else:
                self.gbi_path = "model.safetensors"
        
        if isinstance(self.strategy, str):
            self.strategy = MemoryStrategy(self.strategy.lower())

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v.value if isinstance(v, Enum) else v 
            for k, v in self.__dict__.items()
        }

    def save_pretrained(self, save_directory: str):
        import os
        import json
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "lema_config.json")
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        import os
        import json
        config_file = os.path.join(load_directory, "lema_config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found in {load_directory}")
        
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Handle enum conversion
        if "strategy" in config_dict and isinstance(config_dict["strategy"], str):
            config_dict["strategy"] = MemoryStrategy(config_dict["strategy"])
            
        return cls(**config_dict)

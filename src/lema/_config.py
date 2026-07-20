from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MemoryStrategy(Enum):
    STREAMING = "streaming"
    RESIDENT = "resident"


@dataclass
class LemaConfig:
    model_name_or_path: str
    model_type: str | None = None
    gbi_path: str | None = None
    device: str = "cuda"
    strategy: MemoryStrategy = MemoryStrategy.STREAMING
    backend: str = "auto"
    max_ram_gb: float = 0.0
    max_vram_gb: float = 0.0
    vram_fraction: float = 0.8
    prefetch_distance: int = 2
    lora_rank: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_seq_length: int = 512
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    save_steps: int = 500
    output_dir: str = "output"
    dtype: str = "float16"
    attn_implementation: str = "eager"

    def __post_init__(self):
        if self.gbi_path is None:
            if self.model_name_or_path.endswith(".safetensors"):
                self.gbi_path = self.model_name_or_path
            else:
                self.gbi_path = "model.safetensors"
        if isinstance(self.strategy, str):
            self.strategy = MemoryStrategy(self.strategy.lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }

    def save_pretrained(self, save_directory: str):
        import os, json
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "lema_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        import os, json
        config_file = os.path.join(load_directory, "lema_config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found in {load_directory}")
        with open(config_file) as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        if "strategy" in config_dict and isinstance(config_dict["strategy"], str):
            config_dict["strategy"] = MemoryStrategy(config_dict["strategy"])
        return cls(**config_dict)

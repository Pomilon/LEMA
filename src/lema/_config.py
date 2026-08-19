from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MemoryStrategy(Enum):
    STREAMING = "streaming"
    RESIDENT = "resident"


class TrainingMode(Enum):
    LORA = "lora"
    SELECTIVE_FULL = "selective_full"


class StateStrategy(Enum):
    STREAMING = "streaming"
    VRAM = "vram"


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
    training_mode: TrainingMode = TrainingMode.LORA
    state_strategy: StateStrategy = StateStrategy.STREAMING
    trainable_modules: list[str] = field(default_factory=list)
    trainable_layers: list[str] = field(default_factory=list)
    grad_accum_backend: str = "auto"
    save_optimizer: bool = True
    weight_decay: float = 0.01
    weights_vram: str = "auto"
    opt_state_vram: str = "auto"
    grad_acc_vram: str = "auto"
    kv_vram: str = "auto"
    target_step_time_ms: float = 0.0
    target_tokens_per_sec: float = 0.0
    kv_chunk_size: int = 8192

    def __post_init__(self):
        if self.gbi_path is None:
            if self.model_name_or_path.endswith(".safetensors"):
                self.gbi_path = self.model_name_or_path
            else:
                self.gbi_path = "model.safetensors"
        if isinstance(self.strategy, str):
            self.strategy = MemoryStrategy(self.strategy.lower())
        if isinstance(self.training_mode, str):
            self.training_mode = TrainingMode(self.training_mode.lower())
        if isinstance(self.state_strategy, str):
            self.state_strategy = StateStrategy(self.state_strategy.lower())
        self._validate_full_ft()

    def _validate_full_ft(self):
        if self.grad_accum_backend not in ("auto", "ram", "disk"):
            raise ValueError(
                f"Invalid grad_accum_backend: {self.grad_accum_backend!r}. "
                "Choose 'auto', 'ram', or 'disk'."
            )
        for spec in self.trainable_layers:
            if spec.startswith("last:"):
                k = int(spec.split(":", 1)[1])
                if k <= 0:
                    raise ValueError(
                        f"Invalid trainable_layers spec {spec!r}: 'last:K' requires K >= 1 "
                        "(K=0 silently selects the whole model)."
                    )
            elif spec.startswith("first:"):
                k = int(spec.split(":", 1)[1])
                if k <= 0:
                    raise ValueError(
                        f"Invalid trainable_layers spec {spec!r}: 'first:K' requires K >= 1 "
                        "(K<=0 silently selects nothing or the whole model)."
                    )

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

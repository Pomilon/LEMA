from .base import LemaModelAdapter
from .llama import LlamaAdapter
from .gpt2 import GPT2Adapter

_ADAPTER_REGISTRY = {
    "llama": LlamaAdapter,
    "gpt2": GPT2Adapter
}

def get_adapter(model_type: str, config: dict) -> LemaModelAdapter:
    if model_type not in _ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(_ADAPTER_REGISTRY.keys())}")
    return _ADAPTER_REGISTRY[model_type](config)

def register_adapter(model_type: str, adapter_class: type):
    _ADAPTER_REGISTRY[model_type] = adapter_class

from ._base import LemaModelAdapter
from ._llama import LlamaAdapter
from ._gpt2 import GPT2Adapter
from ._mistral import MistralAdapter
from ._mixtral import MixtralAdapter
from ._lfm2 import Lfm2Adapter

_ADAPTER_REGISTRY: dict[str, type[LemaModelAdapter]] = {
    cls.MODEL_TYPE: cls for cls in [
        LlamaAdapter, GPT2Adapter, MistralAdapter, MixtralAdapter, Lfm2Adapter
    ]
}


def get_adapter(model_type: str, config: dict) -> LemaModelAdapter:
    if model_type not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    return _ADAPTER_REGISTRY[model_type](config)


def register_adapter(model_type: str, adapter_class: type[LemaModelAdapter]):
    _ADAPTER_REGISTRY[model_type] = adapter_class

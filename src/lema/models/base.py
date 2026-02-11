from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

class LemaModelAdapter(ABC):
    """
    Abstract Base Class for LEMA Model Adapters.
    
    This class defines the interface that any model architecture must implement
    to be compatible with the LEMA (Layer-wise Efficient Memory Abstraction) framework.
    It bridges the gap between the raw binary weights managed by LEMA and the 
    PyTorch execution semantics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_layer_metadata(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries, where each dictionary describes a logical "layer"
        or "block" in the model that LEMA should manage as a unit.
        
        Returns:
            List[Dict]: e.g. [{'id': 0, 'name': 'transformer.h.0', 'inputs': [...], 'outputs': [...]}, ...]
        """
        pass

    @abstractmethod
    def construct_layer_module(self, layer_id: int, weights: Dict[str, torch.Tensor], lora_manager: Optional[Any] = None) -> nn.Module:
        """
        Constructs a PyTorch nn.Module for the specified layer using the provided weights.
        The weights will be on the target device (VRAM) when passed here.
        
        Args:
            layer_id (int): The index of the layer to construct.
            weights (Dict[str, torch.Tensor]): A dictionary mapping parameter names to tensors.
            lora_manager (Optional[Any]): The LoRAManager instance to apply adapters.
            
        Returns:
            nn.Module: The executable layer module.
        """
        pass

    @abstractmethod
    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        """
        Executes the forward pass for a single layer.
        
        Args:
            layer_module (nn.Module): The module constructed by construct_layer_module.
            inputs (Any): The input activations (tensor or tuple of tensors).
            **kwargs: Additional arguments (e.g., attention masks, rotary embeddings).
            
        Returns:
            Any: The output activations.
        """
        pass

    @abstractmethod
    def get_param_names_for_layer(self, layer_id: int) -> List[str]:
        """
        Returns the list of parameter names (as found in the safetensors file) 
        required for the specified layer.
        
        Args:
            layer_id (int): Layer index.
            
        Returns:
            List[str]: List of parameter keys.
        """
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Returns the model's hidden size for buffer allocation."""
        pass

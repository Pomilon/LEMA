import torch
from transformers import MistralForCausalLM, MistralConfig, MixtralForCausalLM, MixtralConfig

mistral_config = MistralConfig(num_hidden_layers=1)
mistral = MistralForCausalLM(mistral_config)
print("Mistral keys:", [k for k, _ in mistral.named_parameters()])

mixtral_config = MixtralConfig(num_hidden_layers=1, num_local_experts=2)
mixtral = MixtralForCausalLM(mixtral_config)
print("Mixtral keys:", [k for k, _ in mixtral.named_parameters()])

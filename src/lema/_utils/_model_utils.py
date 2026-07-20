from __future__ import annotations

import torch
import gc
import os
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def break_shared_weights(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    return model


def prepare_monolithic_safetensors(
    model_name_or_path: str,
    output_path: str,
    device: str = "cpu",
    cache_dir: str | None = None,
):
    if cache_dir is None and os.path.exists("/kaggle/working"):
        cache_dir = "/tmp/huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)
    print(f"Loading {model_name_or_path} for monolithic conversion (using {device})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map=device, cache_dir=cache_dir
    )
    model = break_shared_weights(model)
    print(f"Saving monolithic safetensors to {output_path}...")
    sd = model.state_dict()
    save_file(sd, output_path)
    del sd, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

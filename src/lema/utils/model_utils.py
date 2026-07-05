import torch
import os
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoConfig

def break_shared_weights(model: torch.nn.Module):
    """
    Ensures that shared weights (like lm_head and embed_tokens) are distinct copies.
    Required for safetensors compatibility.
    """
    if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            # Only clone the specific shared tensor, not the whole model
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    return model

def prepare_monolithic_safetensors(model_name_or_path: str, output_path: str, device: str = "cpu", cache_dir: str = None):
    """
    Downloads a model and saves it as a single, framework-compatible safetensors file.
    Defaulting to 'cpu' for conversion is safer for various GPU architectures (e.g. P100 vs T4).
    """
    if cache_dir is None and os.path.exists("/kaggle/working"):
        # In Kaggle, use /tmp to avoid filling up the working directory (which has limited space and is for output)
        cache_dir = "/tmp/huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Setting default Kaggle cache directory to {cache_dir}")

    print(f"Loading {model_name_or_path} for monolithic conversion (using {device})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device,
        cache_dir=cache_dir
    )
    model = break_shared_weights(model)
    
    print(f"Saving monolithic safetensors to {output_path}...")
    # Pass state_dict directly to save_file to avoid memory doubling
    sd = model.state_dict()
    save_file(sd, output_path)
    
    # Cleanup
    del sd
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
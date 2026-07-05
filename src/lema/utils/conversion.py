import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from .logger import logger

def convert_to_monolith(model_path: str, output_path: str):
    """
    Finds all .safetensors files in model_path and merges them into a single LEMA monolith.
    """
    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        # Already a single file, just copy or link if needed, but for safety we re-save
        logger.info(f"Source is already a single safetensors file: {model_path}")
        return model_path

    # Find all safetensors files
    files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    files.sort()
    logger.info(f"Merging {len(files)} safetensors files from {model_path} into {output_path}...")

    merged_weights = {}
    for f in tqdm(files, desc="Merging Safetensors"):
        file_path = os.path.join(model_path, f)
        weights = load_file(file_path, device="cpu")
        merged_weights.update(weights)

    logger.info(f"Saving monolith to {output_path} ({len(merged_weights)} tensors)...")
    save_file(merged_weights, output_path)
    logger.info("Conversion complete.")
    
    return output_path

from __future__ import annotations

import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from ._logger import logger


def convert_to_monolith(model_path: str, output_path: str) -> str:
    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        logger.info(f"Source is already a single safetensors file: {model_path}")
        return model_path
    files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    files.sort()
    logger.info(f"Merging {len(files)} safetensors files from {model_path} into {output_path}...")
    merged = {}
    for f in tqdm(files, desc="Merging Safetensors"):
        merged.update(load_file(os.path.join(model_path, f), device="cpu"))
    logger.info(f"Saving monolith to {output_path} ({len(merged)} tensors)...")
    save_file(merged, output_path)
    logger.info("Conversion complete.")
    return output_path

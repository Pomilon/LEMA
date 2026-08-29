#!/usr/bin/env python3
import argparse
import os
import glob
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from lema._quant_backend import quantize_tensor_with_backend


def quantize_checkpoint(model_path: str, output_dir: str, backend: str = "custom", bits: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    files = []
    if os.path.isdir(model_path):
        files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors in {model_path}")
    else:
        files = [model_path]

    tensors = {}
    scales = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                t = handle.get_tensor(key)
                if t.dtype in (torch.int8, torch.int64, torch.int32) or t.dtype == torch.uint8:
                    tensors[key] = t.contiguous()
                    continue
                if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    tensors[key] = t.contiguous()
                    continue
                if t.ndim == 0:
                    tensors[key] = t.contiguous()
                    continue
                q, scale = quantize_tensor_with_backend(t, bits, backend=backend)
                if bits == 4:
                    from lema._quant import pack_int4
                    q = pack_int4(q)
                tensors[key] = q.contiguous()
                scales[f"{key}.scale"] = scale.contiguous().float()

    out_path = os.path.join(output_dir, "model.safetensors")
    combined = {**tensors, **scales}
    save_file(combined, out_path)
    print(f"Quantized {len(tensors)} tensors ({bits}-bit via {backend}) -> {out_path} ({sum(v.numel()*v.element_size() for v in combined.values())/1e6:.1f} MB, scales {len(scales)})")
    # copy config if exists
    for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(model_path, name) if os.path.isdir(model_path) else None
        if src and os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(output_dir, name))
            print(f"Copied {name}")
    # also copy other jsons
    if os.path.isdir(model_path):
        for p in glob.glob(os.path.join(model_path, "*.json")):
            base = os.path.basename(p)
            if base not in ("config.json", "generation_config.json"):
                import shutil
                dst = os.path.join(output_dir, base)
                if not os.path.exists(dst):
                    shutil.copy(p, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Offline quantize checkpoint to int8/int4 + scales for LEMA pre-quantized GBI")
    ap.add_argument("model_path", help="HF model id or local path with safetensors")
    ap.add_argument("--backend", default="custom", choices=["custom", "torchao", "quanto", "bitsandbytes", "auto"])
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    quantize_checkpoint(args.model_path, args.output_dir, args.backend, args.bits)

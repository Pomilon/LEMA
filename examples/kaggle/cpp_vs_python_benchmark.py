"""
LEMA C++ Backend vs Pure Python Backend Benchmark.

Isolates and measures performance of each backend operation:
  1. pack_layer_to_ram     (CPU memcpy)
  2. async_transfer_to_vram (GPU transfer + sync)
  3. End-to-end training loop (C++ ON vs OFF)

Run on Kaggle with GPU (P100/T4) for meaningful results.
"""
import gc
import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, "src")
from lema.config import LemaConfig, MemoryStrategy
from lema.core.gbi import GlobalBinaryIndex
from lema.core.lora import LoRAManager
from lema.engine.trainer import LemaTrainer
from lema.utils.logger import logger
from lema.models.gpt2 import GPT2Adapter
from lema.core.memory import TripleBufferManager, HAS_CPP_BACKEND


def benchmark_packing(adapter, gbi, num_warmup=5, num_iters=30):
    """Benchmark pack_layer_to_ram: Python copy_() vs C++ std::memcpy.

    Measures JUST the memcpy/pack time (disk I/O is excluded since we load
    tensors from GBI once and reuse them).
    """
    layers = adapter.get_layer_metadata()
    dtype = torch.float32
    max_params = 0
    for layer in layers:
        names = adapter.get_param_names_for_layer(layer['id'])
        current = 0
        for name in names:
            shape = gbi.get_tensor_shape(name)
            current += torch.Size(shape).numel()
        max_params = max(max_params, current)

    # Preload all tensors from GBI (exclude disk I/O from measurement)
    layer_tensors = []
    for layer in layers:
        param_names = adapter.get_param_names_for_layer(layer['id'])
        weights = gbi.load_tensors(param_names, device="cpu")
        layer_tensors.append(list(weights.values()))

    results = {"python_pack": [], "cpp_pack": []}

    # --- Python path ---
    print(f"  Python packing: running {num_warmup + num_iters} iters...", end=" ", flush=True)
    for it in range(num_warmup + num_iters):
        buf = torch.empty(max_params, device="cpu", dtype=dtype)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for tensors in layer_tensors:
            offset = 0
            for w in tensors:
                numel = w.numel()
                buf[offset : offset + numel].copy_(w.view(-1))
                offset += numel
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        if it >= num_warmup:
            results["python_pack"].append((t_end - t_start) * 1000)
        del buf
    print("done")

    # --- C++ path ---
    if HAS_CPP_BACKEND:
        from lema.csrc import _lema_cpp
        print(f"  C++   packing: running {num_warmup + num_iters} iters...", end=" ", flush=True)
        # Create C++ manager once (amortize construction overhead)
        cpp_mgr = _lema_cpp.LemaMemoryManager(len(layers) + 2, max_params)
        buf = torch.empty(max_params, device="cpu", dtype=dtype)
        if torch.cuda.is_available():
            buf = buf.pin_memory()
        # Register slot 0 as a catch-all streaming buffer
        cpp_mgr.register_ram_buffer(0, buf)
        cpp_mgr.register_vram_slot(0, torch.empty(1, device="cpu"))

        packed_tensors = []
        for tensors in layer_tensors:
            packed_tensors.append(tensors)

        for it in range(num_warmup + num_iters):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for tensors in packed_tensors:
                cpp_mgr.pack_layer_to_ram(0, tensors)
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            if it >= num_warmup:
                results["cpp_pack"].append((t_end - t_start) * 1000)

        del cpp_mgr, buf, packed_tensors
        print("done")
    else:
        results["cpp_pack"] = [float('inf')] * num_iters

    stats = {}
    for backend in ["python_pack", "cpp_pack"]:
        arr = np.array(results[backend])
        stats[backend] = {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }
    return stats


def benchmark_transfer(max_params, dtype, num_warmup=10, num_iters=100):
    """Benchmark async_transfer_to_vram + synchronization.

    Measures:
      Python: torch.cuda.stream + stream.synchronize()
      C++:    CUDA events + cudaEventSynchronize()
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    results = {"python_xfer_sync": [], "cpp_xfer_sync": []}

    # Shared RAM buffer
    ram_buf = torch.randn(max_params, device="cpu", dtype=dtype).pin_memory()

    # --- Python path ---
    py_vram_buf = torch.empty(max_params, device="cuda", dtype=dtype)
    py_stream = torch.cuda.Stream()

    print(f"  Python transfer+sync: running {num_warmup + num_iters} iters...", end=" ", flush=True)
    for it in range(num_warmup + num_iters):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.cuda.stream(py_stream):
            py_vram_buf[:ram_buf.numel()].copy_(ram_buf, non_blocking=True)
        py_stream.synchronize()
        t_end = time.perf_counter()
        if it >= num_warmup:
            results["python_xfer_sync"].append((t_end - t_start) * 1000)
    del py_vram_buf, py_stream
    print("done")

    # --- C++ path ---
    if HAS_CPP_BACKEND:
        from lema.csrc import _lema_cpp
        cpp_mgr = _lema_cpp.LemaMemoryManager(2, max_params)
        cpp_vram_buf = torch.empty(max_params, device="cuda", dtype=dtype)
        cpp_mgr.register_ram_buffer(0, ram_buf)
        cpp_mgr.register_vram_slot(0, cpp_vram_buf)

        print(f"  C++   transfer+sync: running {num_warmup + num_iters} iters...", end=" ", flush=True)
        for it in range(num_warmup + num_iters):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            event_id = cpp_mgr.async_transfer_to_vram(0, 0)
            cpp_mgr.wait_vram_transfer(event_id)
            t_end = time.perf_counter()
            if it >= num_warmup:
                results["cpp_xfer_sync"].append((t_end - t_start) * 1000)
        del cpp_mgr, cpp_vram_buf
        print("done")
    else:
        results["cpp_xfer_sync"] = [float('inf')] * num_iters

    del ram_buf
    gc.collect()
    torch.cuda.empty_cache()

    stats = {}
    for backend in results:
        arr = np.array(results[backend])
        stats[backend] = {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }
    return stats


def benchmark_training_loop(tmp_dir, use_cpp_backend: bool):
    """End-to-end training loop benchmark with C++ on/off."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from safetensors.torch import save_file

    # Force C++ backend on/off
    import lema.core.memory as mem_module
    old_status = mem_module.HAS_CPP_BACKEND
    mem_module.HAS_CPP_BACKEND = use_cpp_backend and HAS_CPP_BACKEND

    try:
        config_hf = GPT2Config(
            vocab_size=1000, n_positions=128, n_embd=256,
            n_layer=8, n_head=8, attn_implementation="eager"
        )
        model_hf = GPT2LMHeadModel(config_hf)
        state_dict = {k: v.clone().detach() for k, v in model_hf.state_dict().items()}

        model_path = os.path.join(tmp_dir, "bench_model.safetensors")
        save_file(state_dict, model_path)
        config_hf.save_pretrained(tmp_dir)  # So AutoConfig can find it
        del model_hf

        lema_config = LemaConfig(
            model_name_or_path=tmp_dir,
            model_type="gpt2",
            gbi_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            strategy=MemoryStrategy.STREAMING,
            lora_rank=4,
            lora_target_modules=["c_attn"],
            prefetch_distance=2,
        )

        from lema.core.model import LemaModel
        model = LemaModel(lema_config)
        model.initialize_lora()
        optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=0.01)
        trainer = model.get_trainer(optimizer)

        input_ids = torch.randint(0, 1000, (1, 128))
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Warmup
        trainer.train_step(input_ids)

        # Timed steps
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        num_steps = 10
        for _ in range(num_steps):
            trainer.train_step(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t_start

        return (t_elapsed / num_steps) * 1000  # ms/step
    finally:
        mem_module.HAS_CPP_BACKEND = old_status


def run_all_benchmarks(tmp_dir):
    """Run all benchmarks and print comparison. tmp_dir is a path string."""
    print("=" * 70)
    print("LEMA C++ Backend vs Pure Python Backend Benchmark")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    print(f"HAS_CPP_BACKEND: {HAS_CPP_BACKEND}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Setup minimal GPT2 adapter + GBI for benchmarks
    from transformers import GPT2Config
    config_hf = GPT2Config(
        vocab_size=1000, n_positions=128, n_embd=128,
        n_layer=8, n_head=8, attn_implementation="eager"
    )
    from safetensors.torch import save_file
    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel(config_hf)
    state_dict = {k: v.clone().detach() for k, v in model_hf.state_dict().items()}
    model_path = os.path.join(tmp_dir, "pack_bench.safetensors")
    save_file(state_dict, model_path)
    config_hf.save_pretrained(tmp_dir)
    del model_hf

    adapter = GPT2Adapter(config_hf.to_dict())
    gbi = GlobalBinaryIndex(model_path)

    # =========================================================
    # 1. Packing Benchmark
    # =========================================================
    print("\n[1/3] Pack Layer to RAM Benchmark")
    print("    Measures: memcpy of ALL layer params into flat buffer")
    print("-" * 60)
    pack_stats = benchmark_packing(adapter, gbi)

    for backend in ["python_pack", "cpp_pack"]:
        s = pack_stats[backend]
        label = "Python" if "python" in backend else "C++"
        if s["mean_ms"] == float('inf'):
            print(f"  {label:8s}: N/A (backend not available)")
        else:
            print(f"  {label:8s}: {s['mean_ms']:8.3f} ms/step ± {s['std_ms']:6.3f}  "
                  f"[min={s['min_ms']:.3f}, max={s['max_ms']:.3f}]")

    if pack_stats["python_pack"]["mean_ms"] < float('inf') and pack_stats["cpp_pack"]["mean_ms"] < float('inf'):
        py_m = pack_stats["python_pack"]["mean_ms"]
        cpp_m = pack_stats["cpp_pack"]["mean_ms"]
        if py_m > 0:
            ratio = (cpp_m - py_m) / py_m * 100
            faster = "Python" if py_m < cpp_m else "C++"
            print(f"  >>> {faster} is {abs(ratio):.1f}% faster for packing")

    # =========================================================
    # 2. Transfer Benchmark
    # =========================================================
    if torch.cuda.is_available() and HAS_CPP_BACKEND:
        print("\n[2/3] Async Transfer + Sync Benchmark")
        print("    Measures: pinned-RAM → GPU VRAM transfer + synchronization")
        print("-" * 60)

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        sizes = [
            # Sizes matching real LLM decoder layer weights (fp32 buffers):
            # Single attention proj (~33MB for q_proj in fp16)
            ("  32 MB",  8 * 1024 * 1024),
            # Full attention (q+k+v+o, ~132MB in fp16)
            (" 128 MB", 32 * 1024 * 1024),
            # Full decoder layer (attn+mlp, ~400MB in fp16)
            (" 512 MB", 128 * 1024 * 1024),
        ]
        if vram_gb > 4: sizes.append(("1024 MB", 256 * 1024 * 1024))
        if vram_gb > 8: sizes.append(("2048 MB", 512 * 1024 * 1024))

        for label, nparams in sizes:
            tf_stats = benchmark_transfer(nparams, torch.float32)

            for backend in ["python_xfer_sync", "cpp_xfer_sync"]:
                s = tf_stats[backend]
                bg = "Python" if "python" in backend else "C++"
                print(f"  {label} {bg:8s}: {s['mean_ms']:8.3f} ms ± {s['std_ms']:6.3f}")

            if "python_xfer_sync" in tf_stats and "cpp_xfer_sync" in tf_stats:
                py = tf_stats["python_xfer_sync"]["mean_ms"]
                cpp = tf_stats["cpp_xfer_sync"]["mean_ms"]
                if py > 0:
                    ratio = (cpp - py) / py * 100
                    faster = "Python" if py < cpp else "C++"
                    print(f"           >>> {faster} is {abs(ratio):.1f}% faster (total xfer+sync)")
                print()
    else:
        print("\n[2/3] Transfer: SKIPPED (CUDA or C++ backend not available)")

    # =========================================================
    # 3. End-to-End Training Loop
    # =========================================================
    if torch.cuda.is_available():
        print("[3/3] End-to-End Training Loop (GPT2 4-layer, streaming)")
        print("    Measures: full train_step (prefetch + xfer + fwd + bwd)")
        print("-" * 60)

        results = {}
        for use_cpp, label in [(False, "C++ OFF"), (True, "C++ ON ")]:
            if use_cpp and not HAS_CPP_BACKEND:
                continue
            avg_ms = benchmark_training_loop(tmp_dir, use_cpp_backend=use_cpp)
            results[label] = avg_ms
            print(f"  {label}: {avg_ms:.3f} ms/step")

        if "C++ OFF" in results and "C++ ON " in results:
            off = results["C++ OFF"]
            on_ = results["C++ ON "]
            ratio = (on_ - off) / off * 100
            faster = "C++ OFF" if off < on_ else "C++ ON"
            print(f"  >>> {faster} is {abs(ratio):.1f}% faster end-to-end")
            if on_ < off:
                print(f"  >>> C++ backend improvement: {-ratio:.1f}% speedup")
            else:
                print(f"  >>> C++ overhead: {ratio:.1f}% (C++ adds latency)")
    else:
        print("[3/3] Training Loop: SKIPPED (CUDA not available)")

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)

    # Return numeric summary
    result = {}
    try:
        for name in ["python_pack", "cpp_pack"]:
            if name in pack_stats:
                result[name] = pack_stats[name]["mean_ms"]
    except: pass
    return result


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            run_all_benchmarks(td)
        except Exception as e:
            import traceback
            print(f"\nBenchmark error: {e}")
            traceback.print_exc()

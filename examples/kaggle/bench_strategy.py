"""
RESIDENT vs STREAMING Strategy Benchmark.

RESIDENT: all weights loaded into pinned RAM,  no disk I/O during training.
STREAMING: weights loaded on-demand from disk, less RAM usage.
Shows the throughput/VRAM trade-off between the two.
"""
import gc
import time
import torch

from lema import LemaConfig, LemaModel, MemoryStrategy


def get_vram() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def benchmark_strategy(hf_id: str, strategy: MemoryStrategy, seq_len: int = 512, num_steps: int = 10) -> dict:
    label = strategy.value
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  {label}...", end=" ", flush=True)

    try:
        t0 = time.perf_counter()
        config = LemaConfig(
            model_name_or_path=hf_id,
            strategy=strategy,
            lora_rank=16,
            gradient_checkpointing=True,
            prefetch_distance=2,
        )
        model = LemaModel(config)
        model.initialize_lora()
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
        trainer = model.get_trainer(optimizer)
        load_time = time.perf_counter() - t0

        input_ids = torch.randint(0, 1000, (1, seq_len)).cuda()
        trainer.train_step(input_ids, labels=input_ids)
        torch.cuda.synchronize()
        vram_used = get_vram()

        times = []
        for _ in range(num_steps):
            t0 = time.perf_counter()
            trainer.train_step(input_ids, labels=input_ids)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        print(f"OK  — load={load_time:.1f}s, step={avg:.0f}ms, vram={vram_used:.1f}GB")
        return {"load_s": load_time, "step_ms": avg, "vram_gb": vram_used}
    except Exception as e:
        print(f"ERR — {e}")
        return {"load_s": 0, "step_ms": float("inf"), "vram_gb": 0}
    finally:
        try:
            del model, trainer, optimizer
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def main():
    hf_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("=" * 60)
    print("Strategy Benchmark — TinyLlama 1.1B")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"Total VRAM: {total / 1024**3:.1f} GB")
    print("=" * 60)

    print("\n[Loading + Benchmarking]")
    res = benchmark_strategy(hf_id, MemoryStrategy.RESIDENT)
    stream = benchmark_strategy(hf_id, MemoryStrategy.STREAMING)

    print()
    print("=" * 60)
    print(f"{'Strategy':>10} | {'Load(s)':>7} | {'Step(ms)':>8} | {'VRAM(GB)':>8}")
    print("-" * 60)
    for name, r in [("RESIDENT", res), ("STREAMING", stream)]:
        t = f"{r['step_ms']:.0f}" if r['step_ms'] != float("inf") else "ERR"
        v = f"{r['vram_gb']:.1f}" if r['vram_gb'] else "ERR"
        print(f"{name:>10} | {r['load_s']:>7.1f} | {t:>8} | {v:>8}")
    print("=" * 60)

    if res["step_ms"] != float("inf") and stream["step_ms"] != float("inf"):
        ratio = stream["step_ms"] / res["step_ms"]
        print(f"STREAMING step time: {ratio:.1f}x of RESIDENT")
        vram_diff = res["vram_gb"] - stream["vram_gb"]
        if vram_diff > 0.1:
            print(f"STREAMING saves {vram_diff:.1f} GB VRAM (benefit grows with model size)")
        else:
            print(f"VRAM similar for this model size — run on 7B+ to see the advantage")


if __name__ == "__main__":
    main()

"""
LEMA vs Hugging Face CPU Offload Benchmark.

Compares LEMA STREAMING against accelerate's device_map="auto" with CPU offload.
Both solve the same problem (fitting large models in limited VRAM)
but use different approaches.
"""
import gc
import time
import torch

from lema import LemaConfig, LemaModel, MemoryStrategy


def get_vram() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def benchmark_cpu_offload(hf_id: str, seq_len: int = 512, num_steps: int = 5) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  CPU offload...", end=" ", flush=True)

    try:
        from transformers import AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig

        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16,
            device_map="auto", offload_folder="/tmp/cpu_offload",
        )
        model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, 100, (1, seq_len)).cuda()

        model(input_ids, labels=input_ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        vram_used = get_vram()

        times = []
        for _ in range(num_steps):
            t0 = time.perf_counter()
            model(input_ids, labels=input_ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        print(f"OK  — {avg:.0f}ms/step, {vram_used:.1f}GB")
        return {"step_ms": avg, "vram_gb": vram_used}
    except torch.cuda.OutOfMemoryError:
        print("OOM")
        return {"step_ms": float("inf"), "vram_gb": float("inf")}
    except Exception as e:
        print(f"ERR — {e}")
        return {"step_ms": float("inf"), "vram_gb": float("inf")}
    finally:
        try:
            del model, optimizer, input_ids
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def benchmark_lema_streaming(hf_id: str, seq_len: int = 512, num_steps: int = 5) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  LEMA streaming...", end=" ", flush=True)

    try:
        config = LemaConfig(
            model_name_or_path=hf_id,
            strategy=MemoryStrategy.STREAMING,
            lora_rank=16,
            gradient_checkpointing=True,
            prefetch_distance=2,
        )
        model = LemaModel(config)
        model.initialize_lora()
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
        trainer = model.get_trainer(optimizer)

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
        print(f"OK  — {avg:.0f}ms/step, {vram_used:.1f}GB")
        return {"step_ms": avg, "vram_gb": vram_used}
    except torch.cuda.OutOfMemoryError:
        print("OOM")
        return {"step_ms": float("inf"), "vram_gb": float("inf")}
    except Exception as e:
        print(f"ERR — {e}")
        return {"step_ms": float("inf"), "vram_gb": float("inf")}
    finally:
        try:
            del model, trainer, optimizer, input_ids
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def main():
    hf_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    seq_len = 512

    print("=" * 60)
    print("LEMA vs CPU Offload — TinyLlama 1.1B")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    _, total = torch.cuda.mem_get_info()
    print(f"Total VRAM: {total / 1024**3:.1f} GB")
    print("=" * 60)

    print("\n[Benchmarking — single step, seq=512]")
    cpu = benchmark_cpu_offload(hf_id, seq_len)
    lema = benchmark_lema_streaming(hf_id, seq_len)

    print()
    print("=" * 60)
    print(f"{'Method':>15} | {'Step(ms)':>8} | {'VRAM(GB)':>8}")
    print("-" * 60)
    for name, r in [("CPU offload", cpu), ("LEMA streaming", lema)]:
        t = f"{r['step_ms']:.0f}" if r['step_ms'] != float("inf") else "OOM"
        v = f"{r['vram_gb']:.1f}" if r['vram_gb'] != float("inf") else "OOM"
        print(f"{name:>15} | {t:>8} | {v:>8}")
    print("=" * 60)

    if cpu["step_ms"] != float("inf") and lema["step_ms"] != float("inf"):
        ratio = lema["step_ms"] / cpu["step_ms"]
        if ratio > 1:
            print(f"CPU offload is {ratio:.1f}x faster (expected — model fits in VRAM)")
        else:
            print(f"LEMA is {1/ratio:.1f}x faster than CPU offload")


if __name__ == "__main__":
    main()

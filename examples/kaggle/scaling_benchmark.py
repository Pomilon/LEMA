"""
LEMA vs PEFT — VRAM & Throughput Scaling Benchmark.

Tests both approaches across increasing batch sizes and sequence lengths
to demonstrate LEMA's headroom advantage at scale.
Includes TinyLlama 1.1B and Llama-2 7B.
"""
import gc
import time
import torch

from lema import LemaConfig, LemaModel, MemoryStrategy


def get_vram() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def benchmark_peft(model_info: dict, batch_sizes: list[int], seq_lens: list[int], num_steps: int = 5) -> dict:
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig

    hf_id, name = model_info["hf_id"], model_info["name"]
    results = {}
    for bs in batch_sizes:
        for seq in seq_lens:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  {name} PEFT bs={bs} seq={seq}...", end=" ", flush=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id, torch_dtype=torch.float16, device_map="cuda",
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
                input_ids = torch.randint(0, 100, (bs, seq)).cuda()

                model(input_ids, labels=input_ids).loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                vram = get_vram()

                times = []
                for _ in range(num_steps):
                    t0 = time.perf_counter()
                    model(input_ids, labels=input_ids).loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)

                avg = sum(times) / len(times)
                results[(bs, seq)] = {"time_ms": avg, "vram_gb": vram}
                print(f"OK  — {avg:.0f}ms, {vram:.1f}GB")
            except torch.cuda.OutOfMemoryError:
                results[(bs, seq)] = {"time_ms": float("inf"), "vram_gb": float("inf")}
                print("OOM")
            except Exception as e:
                results[(bs, seq)] = {"time_ms": float("inf"), "vram_gb": float("inf")}
                print(f"ERR — {e}")
            finally:
                try:
                    del model, optimizer, input_ids
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
    return results


def benchmark_lema(model_info: dict, batch_sizes: list[int], seq_lens: list[int], num_steps: int = 5) -> dict:
    hf_id, name = model_info["hf_id"], model_info["name"]
    results = {}
    for bs in batch_sizes:
        for seq in seq_lens:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  {name} LEMA bs={bs} seq={seq}...", end=" ", flush=True)
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

                input_ids = torch.randint(0, 1000, (bs, seq)).cuda()
                trainer.train_step(input_ids, labels=input_ids)
                torch.cuda.synchronize()
                vram = get_vram()

                times = []
                for _ in range(num_steps):
                    t0 = time.perf_counter()
                    trainer.train_step(input_ids, labels=input_ids)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)

                avg = sum(times) / len(times)
                results[(bs, seq)] = {"time_ms": avg, "vram_gb": vram}
                print(f"OK  — {avg:.0f}ms, {vram:.1f}GB")
            except torch.cuda.OutOfMemoryError:
                results[(bs, seq)] = {"time_ms": float("inf"), "vram_gb": float("inf")}
                print("OOM")
            except Exception as e:
                results[(bs, seq)] = {"time_ms": float("inf"), "vram_gb": float("inf")}
                print(f"ERR — {e}")
            finally:
                try:
                    del model, trainer, optimizer, input_ids
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
    return results


def print_results(name: str, peft: dict, lema: dict, batch_sizes: list[int], seq_lens: list[int]):
    print()
    print("=" * 75)
    print(f"RESULTS: {name}")
    print("=" * 75)
    header = f"{'Seq':>6} | {'Batch':>6} | {'PEFT ms':>8} | {'PEFT VRAM':>9} | {'LEMA ms':>8} | {'LEMA VRAM':>9} | {'Status':>8}"
    print(header)
    print("-" * 75)
    for seq in seq_lens:
        for bs in batch_sizes:
            p = peft.get((bs, seq), {"time_ms": float("inf"), "vram_gb": float("inf")})
            l = lema.get((bs, seq), {"time_ms": float("inf"), "vram_gb": float("inf")})
            pt = f"{p['time_ms']:.0f}" if p['time_ms'] != float("inf") else "  OOM"
            pv = f"{p['vram_gb']:.1f}" if p['vram_gb'] != float("inf") else "  OOM"
            lt = f"{l['time_ms']:.0f}" if l['time_ms'] != float("inf") else "  OOM"
            lv = f"{l['vram_gb']:.1f}" if l['vram_gb'] != float("inf") else "  OOM"
            status = "Both OK" if p['time_ms'] != float("inf") and l['time_ms'] != float("inf") else \
                     "LEMA OK" if l['time_ms'] != float("inf") else \
                     "PEFT OK" if p['time_ms'] != float("inf") else \
                     "Both OOM"
            print(f"{seq:>6} | {bs:>6} | {pt:>8} | {pv:>9} | {lt:>8} | {lv:>9} | {status:>8}")
    print("=" * 75)


def main():
    models = [
        {"name": "TinyLlama 1.1B", "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        {"name": "Llama-2 7B",     "hf_id": "NousResearch/Llama-2-7b-hf"},
    ]
    batch_sizes = [1, 2, 4, 8]
    seq_lens = [128, 256, 512]

    torch.cuda.empty_cache()
    gc.collect()

    _, total = torch.cuda.mem_get_info()
    print("=" * 75)
    print("LEMA vs PEFT — Scaling Benchmarks")
    print(f"GPU: {torch.cuda.get_device_name(0)}, Total VRAM: {total / 1024**3:.1f} GB")
    print(f"Testing: batch={batch_sizes}, seq={seq_lens}")
    print("=" * 75)

    for model_info in models:
        print(f"\n{'=' * 75}")
        print(f"Scaling: {model_info['name']}")
        print(f"{'=' * 75}")

        print(f"\n[PEFT]")
        peft = benchmark_peft(model_info, batch_sizes, seq_lens)

        print(f"\n[LEMA]")
        lema = benchmark_lema(model_info, batch_sizes, seq_lens)

        print_results(model_info["name"], peft, lema, batch_sizes, seq_lens)


if __name__ == "__main__":
    main()

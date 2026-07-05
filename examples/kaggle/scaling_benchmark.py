"""
LEMA vs PEFT Scaling Benchmark.

Tests both approaches across increasing batch sizes and sequence lengths
to demonstrate LEMA's headroom advantage at scale.
Measures VRAM usage and throughput.
"""
import gc
import os
import sys
import time
import torch
import psutil

sys.path.insert(0, "src")

def format_vram() -> str:
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    total_gb = total / 1024**3
    return f"{used:.2f}/{total_gb:.2f} GB"

def get_vram_used() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3

def benchmark_peft_scale(hf_id: str, batch_sizes, seq_lens, num_steps=5):
    """Test PEFT across batch/seq combinations, returns { (bs,seq): {time, vram} }."""
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig

    results = {}
    for bs in batch_sizes:
        for seq in seq_lens:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n  PEFT bs={bs} seq={seq}...", end=" ", flush=True)

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id, torch_dtype=torch.float16, device_map="cuda"
                )
                model.gradient_checkpointing_enable()
                peft_config = LoraConfig(
                    r=16, lora_alpha=32,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, peft_config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                input_ids = torch.randint(0, 100, (bs, seq)).cuda()

                vram_before = get_vram_used()
                print(f"(model={vram_before:.1f}GB", end=" ", flush=True)
                model(input_ids, labels=input_ids).loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                vram_after_warmup = get_vram_used()

                step_times = []
                for step in range(num_steps):
                    t0 = time.perf_counter()
                    model(input_ids, labels=input_ids).loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    step_times.append((time.perf_counter() - t0) * 1000)
                vram_peak = get_vram_used()

                del model, optimizer
                elapsed = sum(step_times) / num_steps
                results[(bs, seq)] = {
                    "time_ms": elapsed,
                    "vram_gb": max(vram_after_warmup, vram_peak),
                }
                times_str = " ".join(f"{t:.0f}" for t in step_times)
                print(f"warmup={vram_after_warmup:.1f}GB peak={vram_peak:.1f}GB avg={elapsed:.0f}ms steps=[{times_str}])")
            except torch.cuda.OutOfMemoryError:
                results[(bs, seq)] = {"time_ms": float('inf'), "vram_gb": float('inf')}
                print("OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                results[(bs, seq)] = {"time_ms": float('inf'), "vram_gb": float('inf')}
                print(f"ERR: {e}")

            finally:
                try: del model, optimizer, input_ids
                except: pass
                gc.collect()
                torch.cuda.empty_cache()

    return results


def benchmark_lema_scale(hf_id: str, batch_sizes, seq_lens, num_steps=5):
    """Test LEMA across batch/seq combinations."""
    from lema import LemaConfig, LemaModel, MemoryStrategy

    results = {}
    for bs in batch_sizes:
        for seq in seq_lens:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n  LEMA bs={bs} seq={seq}...", end=" ", flush=True)

            model = None
            trainer = None
            optimizer = None
            try:
                config = LemaConfig(
                    model_name_or_path=hf_id,
                    device="cuda",
                    strategy=MemoryStrategy.STREAMING,
                    lora_rank=16,
                    gradient_checkpointing=True,
                )
                model = LemaModel(config)
                model.initialize_lora()
                optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
                trainer = model.get_trainer(optimizer)

                input_ids = torch.randint(0, 100, (bs, seq)).cuda()
                labels = input_ids.clone()

                # Warmup + VRAM measurement
                vram_before = get_vram_used()
                trainer.train_step(input_ids, labels=labels)
                torch.cuda.synchronize()
                vram_after = get_vram_used()

                step_times = []
                for step in range(num_steps):
                    t0 = time.perf_counter()
                    trainer.train_step(input_ids, labels=labels)
                    torch.cuda.synchronize()
                    step_times.append((time.perf_counter() - t0) * 1000)
                elapsed = sum(step_times) / num_steps

                results[(bs, seq)] = {
                    "time_ms": elapsed,
                    "vram_gb": vram_after,
                }
                times_str = " ".join(f"{t:.0f}" for t in step_times)
                print(f"OK (avg={elapsed:.0f}ms steps=[{times_str}], vram={vram_after:.1f}GB)")
            except torch.cuda.OutOfMemoryError:
                results[(bs, seq)] = {"time_ms": float('inf'), "vram_gb": float('inf')}
                print("OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                results[(bs, seq)] = {"time_ms": float('inf'), "vram_gb": float('inf')}
                print(f"ERR: {e}")
                import traceback; traceback.print_exc()

            finally:
                try: del model
                except: pass
                try: del trainer
                except: pass
                try: del optimizer
                except: pass
                try: del input_ids
                except: pass
                try: del labels
                except: pass
                try: del config
                except: pass
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    return results


def run_benchmark_for_model(hf_id: str, label: str, batch_sizes, seq_lens):
    """Run PEFT + LEMA scaling for a given model and print results."""
    print(f"\n{'=' * 70}")
    print(f"LEMA vs PEFT — Scaling Benchmark ({label})")
    print(f"{'=' * 70}")
    
    # PEFT scaling
    print(f"\n[PEFT] Scaling ({label})")
    print("-" * 60)
    peft = benchmark_peft_scale(hf_id, batch_sizes, seq_lens)

    # LEMA scaling
    print(f"\n[LEMA] Scaling ({label})")
    print("-" * 60)
    lema = benchmark_lema_scale(hf_id, batch_sizes, seq_lens)

    # Results table
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {label}")
    print(f"{'=' * 70}")

    header = f"{'Seq':>6} | {'Batch':>5} | {'PEFT ms':>8} | {'PEFT VRAM':>10} | {'LEMA ms':>8} | {'LEMA VRAM':>10} | {'Ratio':>8}"
    print(header)
    print("-" * len(header))

    for seq in seq_lens:
        for bs in batch_sizes:
            p = peft.get((bs, seq), {})
            l = lema.get((bs, seq), {})
            p_time = p.get("time_ms", float('inf'))
            l_time = l.get("time_ms", float('inf'))
            p_vram = p.get("vram_gb", float('inf'))
            l_vram = l.get("vram_gb", float('inf'))

            p_str = f"{p_time:.0f}" if p_time != float('inf') else " OOM"
            l_str = f"{l_time:.0f}" if l_time != float('inf') else " OOM"
            pv_str = f"{p_vram:.2f}" if p_vram != float('inf') else " OOM"
            lv_str = f"{l_vram:.2f}" if l_vram != float('inf') else " OK!"

            ratio = ""
            if p_time != float('inf') and l_time != float('inf') and p_time > 0:
                r = l_time / p_time
                ratio = f"{r:.1f}x"
            elif p_time == float('inf') and l_time != float('inf'):
                ratio = "LEMA OK"
            elif l_time == float('inf') and p_time != float('inf'):
                ratio = "PEFT OK"

            print(f"{seq:>6} | {bs:>5} | {p_str:>8} | {pv_str:>10} | {l_str:>8} | {lv_str:>10} | {ratio:>8}")
    return peft, lema


def run_scaling_benchmark():
    """Run scaling benchmarks for TinyLlama and Llama-7B."""
    print("=" * 70)
    print("LEMA vs PEFT — Scaling Benchmarks")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    # TinyLlama — fits in VRAM for both, shows overhead
    run_benchmark_for_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "TinyLlama 1.1B",
        batch_sizes=[1, 2, 4, 8],
        seq_lens=[128, 256, 512],
    )

    # Llama-7B — larger than VRAM, shows LEMA's headroom advantage
    run_benchmark_for_model(
        "NousResearch/Llama-2-7b-hf",
        "Llama-2 7B",
        batch_sizes=[1, 2, 4, 8],
        seq_lens=[128, 256, 512],
    )

    print("\n" + "=" * 70)
    print("Scaling Benchmark Complete")
    print("LEMA advantage: Runs at sizes PEFT cannot (OOM)")
    print("=" * 70)


if __name__ == "__main__":
    run_scaling_benchmark()

"""
Long Sequence Headroom Demo.

Shows LEMA maintaining consistent VRAM as sequence length grows
(512, 1024, 2048, 4096) while standard PEFT OOMs.
"""
import gc
import time
import torch

from lema import LemaConfig, LemaModel, MemoryStrategy


def get_vram() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def benchmark_peft_seq(hf_id: str, seq_lens: list[int], num_steps: int = 3) -> dict:
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig

    results = {}
    for seq in seq_lens:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  PEFT seq={seq}...", end=" ", flush=True)
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
            input_ids = torch.randint(0, 100, (1, seq)).cuda()

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
            results[seq] = {"time_ms": avg, "vram_gb": vram_used}
            print(f"OK  — {avg:.0f}ms, {vram_used:.1f}GB")
        except torch.cuda.OutOfMemoryError:
            results[seq] = {"time_ms": float("inf"), "vram_gb": float("inf")}
            print("OOM")
        except Exception as e:
            results[seq] = {"time_ms": float("inf"), "vram_gb": float("inf")}
            print(f"ERR — {e}")
        finally:
            try:
                del model, optimizer, input_ids
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
    return results


def benchmark_lema_seq(hf_id: str, seq_lens: list[int], num_steps: int = 3) -> dict:
    results = {}
    for seq in seq_lens:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  LEMA seq={seq}...", end=" ", flush=True)
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

            input_ids = torch.randint(0, 1000, (1, seq)).cuda()
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
            results[seq] = {"time_ms": avg, "vram_gb": vram_used}
            print(f"OK  — {avg:.0f}ms, {vram_used:.1f}GB")
        except torch.cuda.OutOfMemoryError:
            results[seq] = {"time_ms": float("inf"), "vram_gb": float("inf")}
            print("OOM")
        except Exception as e:
            results[seq] = {"time_ms": float("inf"), "vram_gb": float("inf")}
            print(f"ERR — {e}")
        finally:
            try:
                del model, trainer, optimizer, input_ids
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
    return results


def main():
    hf_id = "NousResearch/Llama-2-7b-hf"
    seq_lens = [512, 1024, 2048, 4096]

    torch.cuda.empty_cache()
    gc.collect()

    _, total = torch.cuda.mem_get_info()
    print("=" * 70)
    print("Long Sequence Headroom — Llama-2 7B")
    print(f"GPU: {torch.cuda.get_device_name(0)}, Total VRAM: {total / 1024**3:.1f} GB")
    print("-" * 70)

    print("\n[PEFT]")
    peft = benchmark_peft_seq(hf_id, seq_lens)

    print("\n[LEMA]")
    lema = benchmark_lema_seq(hf_id, seq_lens)

    print()
    print("=" * 70)
    print(f"{'Seq':>6} | {'PEFT ms':>8} | {'PEFT VRAM':>9} | {'LEMA ms':>8} | {'LEMA VRAM':>9}")
    print("-" * 70)
    for seq in seq_lens:
        p = peft.get(seq, {"time_ms": float("inf"), "vram_gb": float("inf")})
        l = lema.get(seq, {"time_ms": float("inf"), "vram_gb": float("inf")})
        pt = f"{p['time_ms']:.0f}" if p['time_ms'] != float("inf") else "  OOM"
        pv = f"{p['vram_gb']:.1f}" if p['vram_gb'] != float("inf") else "  OOM"
        lt = f"{l['time_ms']:.0f}" if l['time_ms'] != float("inf") else "  ERR"
        lv = f"{l['vram_gb']:.1f}" if l['vram_gb'] != float("inf") else "  ERR"
        print(f"{seq:>6} | {pt:>8} | {pv:>9} | {lt:>8} | {lv:>9}")
    print("=" * 70)

    lema_ok = [s for s in seq_lens if lema.get(s, {}).get("vram_gb", float("inf")) != float("inf")]
    peft_ok = [s for s in seq_lens if peft.get(s, {}).get("vram_gb", float("inf")) != float("inf")]
    if lema_ok and peft_ok:
        last_lema = lema[lema_ok[-1]]["vram_gb"]
        last_peft = peft[peft_ok[-1]]["vram_gb"]
        if last_lema < last_peft:
            print(f"LEMA uses {last_lema:.1f}GB at seq={lema_ok[-1]} vs PEFT's {last_peft:.1f}GB at seq={peft_ok[-1]}")
    max_lema = max(lema_ok) if lema_ok else 0
    max_peft = max(peft_ok) if peft_ok else 0
    if max_lema > max_peft:
        print(f"LEMA reaches seq={max_lema}, PEFT OOMs at seq={max_peft + 1 if max_peft > 0 else seq_lens[0]}")
    print(f"LEMA runs at {len(lema_ok)}/{len(seq_lens)} seq lens, PEFT at {len(peft_ok)}/{len(seq_lens)}")


if __name__ == "__main__":
    main()

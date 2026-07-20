"""
VRAM Stability Test.

Runs 50+ training steps with a 7B-scale model (or the largest available)
and logs VRAM usage every 10 steps to prove there is no memory leak.
"""
import gc
import time
import torch

from lema import LemaConfig, LemaModel, MemoryStrategy


def get_vram() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def main():
    hf_id = "NousResearch/Llama-2-7b-hf"
    num_steps = 60
    log_interval = 10

    torch.cuda.empty_cache()
    gc.collect()

    _, total = torch.cuda.mem_get_info()
    print("=" * 70)
    print("VRAM Stability Test — Llama-2 7B")
    print(f"GPU: {torch.cuda.get_device_name(0)}, Total VRAM: {total / 1024**3:.1f} GB")
    print(f"Running {num_steps} training steps, logging VRAM every {log_interval}")
    print("=" * 70)

    config = LemaConfig(
        model_name_or_path=hf_id,
        strategy=MemoryStrategy.STREAMING,
        lora_rank=16,
        gradient_checkpointing=True,
        prefetch_distance=2,
    )
    model = LemaModel(config)
    model.initialize_lora()
    optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=0.01)
    trainer = model.get_trainer(optimizer)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    text = "The quick brown fox jumps over the lazy dog. " * 5
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].cuda()
    # Pad to 128 if shorter
    if input_ids.shape[1] < 128:
        pad = torch.full((1, 128 - input_ids.shape[1]), tokenizer.pad_token_id or 0, dtype=torch.long).cuda()
        input_ids = torch.cat([input_ids, pad], dim=1)

    print()
    print(f"{'Step':>6} | {'VRAM(GB)':>8} | {'Step(ms)':>8} | {'Loss':>8}")
    print("-" * 70)

    vram_log = []
    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()
        _, loss = trainer.train_step(input_ids, labels=input_ids)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        if step % log_interval == 0 or step == 1:
            vram = get_vram()
            vram_log.append(vram)
            print(f"{step:>6} | {vram:>8.2f} | {elapsed:>8.0f} | {loss:>8.4f}")

    print("-" * 70)
    print(f"{'Done':>6} | {'':>8} | {'':>8} | {'':>8}")

    del model, trainer, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("VRAM Stability Summary")
    print("=" * 70)
    logged_steps = [1] + list(range(log_interval, num_steps + 1, log_interval))
    for step, vram in zip(logged_steps, vram_log):
        print(f"  Step {step:>4}: {vram:.2f} GB")

    if len(vram_log) >= 2:
        drift = vram_log[-1] - vram_log[0]
        print(f"  Drift over {num_steps} steps: {drift:+.3f} GB")
        if abs(drift) < 0.5:
            print("  ✅ VRAM stable — no leak detected")
        else:
            print(f"  ⚠️  VRAM changed by {drift:+.2f} GB over {num_steps} steps")


if __name__ == "__main__":
    main()

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
from lema import LemaConfig, LemaModel, MemoryStrategy


def build(tmp_path, accum_steps):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"
    hf_cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=2, n_head=2,
                        attn_implementation="eager", tie_word_embeddings=False)
    model = GPT2LMHeadModel(hf_cfg)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(model_path))
    hf_cfg.save_pretrained(str(model_dir))
    lema_cfg = LemaConfig(model_name_or_path=str(model_dir), model_type="gpt2",
                          gbi_path=str(model_path), device="cpu",
                          dtype="float32",
                          strategy=MemoryStrategy.STREAMING, training_mode="selective_full",
                          trainable_modules=["c_attn"], trainable_layers=["last:1"],
                          learning_rate=0.05, weight_decay=0.0,
                          gradient_accumulation_steps=accum_steps,
                          max_ram_gb=64,
                          output_dir=str(tmp_path))
    return LemaModel(lema_cfg)


def test_accumulation_steps_only_applied_at_boundary(tmp_path):
    m = build(tmp_path, accum_steps=3)
    mgr = m.full_ft_manager
    ids = torch.randint(0, 64, (1, 10))
    w0 = {k: v.clone() for k, v in mgr.true_weights.items()}

    # micro-batches 1 and 2 must NOT step
    for _ in range(2):
        m.get_trainer().train_step(ids, labels=ids)
    for k in w0:
        assert torch.allclose(w0[k], mgr.true_weights[k]), "weights changed before accumulation boundary"

    # micro-batch 3 steps
    m.get_trainer().train_step(ids, labels=ids)
    changed = any(not torch.allclose(w0[k], mgr.true_weights[k]) for k in w0)
    assert changed, "weights did not change at accumulation boundary"


def test_accumulators_zeroed_after_step(tmp_path):
    m = build(tmp_path, accum_steps=2)
    mgr = m.full_ft_manager
    ids = torch.randint(0, 64, (1, 10))
    for _ in range(4):
        m.get_trainer().train_step(ids, labels=ids)
    for acc in mgr.accumulators.values():
        assert acc.abs().sum().item() == 0, "accumulator not zeroed after step"

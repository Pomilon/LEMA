import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
from lema import LemaConfig, LemaModel, MemoryStrategy
from lema._full_ft import FullFTManager


def test_selective_full_ft_training_loop(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"

    hf_cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                        attn_implementation="eager", tie_word_embeddings=False)
    model = GPT2LMHeadModel(hf_cfg)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(model_path))
    hf_cfg.save_pretrained(str(model_dir))

    lema_cfg = LemaConfig(
        model_name_or_path=str(model_dir),
        model_type="gpt2",
        gbi_path=str(model_path),
        device="cpu",
        dtype="float32",
        strategy=MemoryStrategy.STREAMING,
        training_mode="selective_full",
        trainable_modules=["c_attn"],
        trainable_layers=["last:1"],
        learning_rate=0.05,
        weight_decay=0.0,
        save_steps=2,
        max_ram_gb=64,
        output_dir=str(tmp_path / "checkpoints"),
    )

    model = LemaModel(lema_cfg)
    manager = model.full_ft_manager
    assert manager is not None
    assert set(manager.selected.keys()) == {3}

    before = {k: v.clone() for k, v in manager.true_weights.items()}
    trainer = model.get_trainer()

    input_ids = torch.randint(0, 64, (1, 10))
    for _ in range(4):
        trainer.train_step(input_ids, labels=input_ids)

    changed = any(not torch.allclose(before[k], manager.true_weights[k]) for k in before)
    assert changed, "selected weights did not change"

    assert (tmp_path / "checkpoints" / "checkpoint-2" / "delta.safetensors").exists()
    assert (tmp_path / "checkpoints" / "checkpoint-4" / "delta.safetensors").exists()


def test_full_ft_manager_created_in_selective_mode(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"
    hf_cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                        attn_implementation="eager", tie_word_embeddings=False)
    model = GPT2LMHeadModel(hf_cfg)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(model_path))
    hf_cfg.save_pretrained(str(model_dir))

    lema_cfg = LemaConfig(model_name_or_path=str(model_dir), model_type="gpt2",
                          gbi_path=str(model_path), device="cpu",
                          dtype="float32",
                          strategy=MemoryStrategy.STREAMING, training_mode="selective_full")
    m = LemaModel(lema_cfg)
    assert isinstance(m.full_ft_manager, FullFTManager)


def test_accumulation_counter_does_not_advance_during_no_grad(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"
    hf_cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                        attn_implementation="eager", tie_word_embeddings=False)
    model = GPT2LMHeadModel(hf_cfg)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(model_path))
    hf_cfg.save_pretrained(str(model_dir))

    lema_cfg = LemaConfig(model_name_or_path=str(model_dir), model_type="gpt2",
                          gbi_path=str(model_path), device="cpu",
                          dtype="float32",
                          strategy=MemoryStrategy.STREAMING, training_mode="selective_full",
                          trainable_modules=["c_attn"], trainable_layers=["last:1"],
                          gradient_accumulation_steps=2,
                          max_ram_gb=64, output_dir=str(tmp_path / "out"))
    m = LemaModel(lema_cfg)
    manager = m.full_ft_manager
    trainer = m.get_trainer()

    input_ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        trainer.train_step(input_ids, labels=input_ids)
    assert manager.accumulation_step == 0

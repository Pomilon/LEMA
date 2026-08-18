import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors import safe_open
from safetensors.torch import save_file
from lema import LemaConfig, LemaModel, MemoryStrategy


def test_whole_model_training_with_tied_embeddings(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"

    hf_cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                        attn_implementation="eager", tie_word_embeddings=True)
    assert hf_cfg.tie_word_embeddings is True
    model = GPT2LMHeadModel(hf_cfg)
    # Tied config shares wte.weight and lm_head.weight -> clone to break shared
    # memory so safetensors can write both keys.
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()}, str(model_path))
    hf_cfg.save_pretrained(str(model_dir))

    lema_cfg = LemaConfig(
        model_name_or_path=str(model_dir),
        model_type="gpt2",
        gbi_path=str(model_path),
        device="cpu",
        dtype="float32",
        strategy=MemoryStrategy.STREAMING,
        training_mode="selective_full",
        learning_rate=0.05,
        weight_decay=0.0,
        save_steps=2,
        max_ram_gb=64,
        output_dir=str(tmp_path / "checkpoints"),
    )

    model = LemaModel(lema_cfg)
    manager = model.full_ft_manager
    assert manager is not None

    selected_names = {name for (_, name) in manager.true_weights}
    assert "transformer.wte.weight" in selected_names
    assert "lm_head.weight" not in selected_names, \
        "tied lm_head.weight must not be selected as a separately-trainable weight"

    before = {k: v.clone() for k, v in manager.true_weights.items()}
    trainer = model.get_trainer()

    input_ids = torch.randint(0, 64, (1, 10))
    for _ in range(4):
        trainer.train_step(input_ids, labels=input_ids)

    changed = any(not torch.allclose(before[k], manager.true_weights[k]) for k in before)
    assert changed, "selected weights did not change"

    delta_path = tmp_path / "checkpoints" / "checkpoint-4" / "delta.safetensors"
    assert delta_path.exists()
    with safe_open(str(delta_path), framework="pt", device="cpu") as f:
        delta_keys = set(f.keys())
    assert "transformer.wte.weight" in delta_keys
    assert "lm_head.weight" not in delta_keys

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file, load_file
from lema._config import LemaConfig, MemoryStrategy
from lema import LemaModel
from lema._utils._conversion import merge_delta
import os


def make_full_model_dir(tmp_path, trained=True, steps=3):
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
                          save_steps=999, max_ram_gb=64, output_dir=str(tmp_path))
    m = LemaModel(lema_cfg)
    if trained:
        trainer = m.get_trainer()
        ids = torch.randint(0, 64, (1, 10))
        for _ in range(steps):
            trainer.train_step(ids, labels=ids)
    return model_dir, model_path, hf_cfg, m


def test_delta_save_and_restore(tmp_path):
    model_dir, model_path, hf_cfg, m = make_full_model_dir(tmp_path, trained=True)
    ckpt = tmp_path / "ckpt"
    m.full_ft_manager.save_delta(str(ckpt))
    assert (ckpt / "delta.safetensors").exists()
    assert (ckpt / "delta.index.json").exists()
    delta = load_file(str(ckpt / "delta.safetensors"))
    assert all("c_attn" in k for k in delta.keys())
    assert all(delta[k].dtype == torch.float32 for k in delta)
    assert any(delta[k].abs().max() > 0 for k in delta)

    m2 = LemaModel(LemaConfig(model_name_or_path=str(model_dir), model_type="gpt2",
                              gbi_path=str(model_path), device="cpu",
                              dtype="float32",
                              strategy=MemoryStrategy.STREAMING,
                              training_mode="selective_full",
                              trainable_modules=["c_attn"], trainable_layers=["last:1"]))
    m2.full_ft_manager.load_delta(str(ckpt))
    for key, w in m.full_ft_manager.true_weights.items():
        assert torch.allclose(w, m2.full_ft_manager.true_weights[key], atol=1e-4)


def test_merge_delta_produces_servable_model(tmp_path):
    model_dir, model_path, hf_cfg, m = make_full_model_dir(tmp_path, trained=True)
    ckpt = tmp_path / "ckpt"
    m.full_ft_manager.save_delta(str(ckpt))
    out = tmp_path / "merged.safetensors"
    merge_delta(str(model_path), str(ckpt / "delta.safetensors"), str(out))
    assert os.path.exists(str(out))

    merged = load_file(str(out))
    base = load_file(str(model_path))
    delta = load_file(str(ckpt / "delta.safetensors"))
    for k in merged:
        expected = base[k] + delta[k] if k in delta else base[k]
        assert torch.allclose(merged[k], expected, atol=1e-5), k


def test_from_pretrained_applies_delta(tmp_path):
    model_dir, model_path, hf_cfg, m = make_full_model_dir(tmp_path, trained=True)
    ckpt = tmp_path / "ckpt"
    m.full_ft_manager.save_delta(str(ckpt))
    m.config.save_pretrained(str(ckpt))
    m2 = LemaModel.from_pretrained(str(ckpt), device="cpu")
    for key, w in m.full_ft_manager.true_weights.items():
        assert torch.allclose(w, m2.full_ft_manager.true_weights[key], atol=1e-4)


def _fresh_manager(model_dir, model_path, tmp_path):
    return LemaModel(LemaConfig(model_name_or_path=str(model_dir), model_type="gpt2",
                                gbi_path=str(model_path), device="cpu",
                                dtype="float32",
                                strategy=MemoryStrategy.STREAMING,
                                training_mode="selective_full",
                                trainable_modules=["c_attn"], trainable_layers=["last:1"],
                                learning_rate=0.05, weight_decay=0.0,
                                max_ram_gb=64, output_dir=str(tmp_path)))


def test_optimizer_state_round_trip(tmp_path):
    model_dir, model_path, hf_cfg, m = make_full_model_dir(tmp_path, trained=True)
    manager = m.full_ft_manager
    assert any(s["exp_avg"].abs().max() > 0 for s in manager.opt_states.values()), \
        "trained model must have nonzero exp_avg moments"

    ckpt = tmp_path / "ckpt"
    manager.save_optimizer(str(ckpt))
    assert (ckpt / "optimizer_fullft.bin").exists()

    m2 = _fresh_manager(model_dir, model_path, tmp_path)
    m2.full_ft_manager.load_optimizer(str(ckpt))
    assert m2.full_ft_manager.layer_steps == manager.layer_steps
    for key, s in manager.opt_states.items():
        assert torch.equal(m2.full_ft_manager.opt_states[key]["exp_avg"], s["exp_avg"]), key
        assert torch.equal(m2.full_ft_manager.opt_states[key]["exp_avg_sq"], s["exp_avg_sq"]), key


def test_optimizer_payload_is_weights_only_safe(tmp_path):
    model_dir, model_path, hf_cfg, m = make_full_model_dir(tmp_path, trained=True)
    ckpt = tmp_path / "ckpt"
    m.full_ft_manager.save_optimizer(str(ckpt))
    data = torch.load(str(ckpt / "optimizer_fullft.bin"), map_location="cpu", weights_only=True)
    assert "layer_steps" in data
    assert "states" in data


def test_load_optimizer_missing_file_is_noop(tmp_path):
    model_dir, model_path, hf_cfg, m = make_full_model_dir(tmp_path, trained=False)
    manager = m.full_ft_manager
    empty = tmp_path / "empty"
    empty.mkdir()
    before_steps = dict(manager.layer_steps)
    before_states = {k: {m: t.clone() for m, t in s.items()} for k, s in manager.opt_states.items()}
    manager.load_optimizer(str(empty))
    assert manager.layer_steps == before_steps
    for key, s in manager.opt_states.items():
        assert torch.equal(s["exp_avg"], before_states[key]["exp_avg"]), key
        assert torch.equal(s["exp_avg_sq"], before_states[key]["exp_avg_sq"]), key

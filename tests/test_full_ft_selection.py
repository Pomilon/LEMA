import torch
from transformers import GPT2Config
from safetensors.torch import save_file
from lema._gbi import GlobalBinaryIndex
from lema.adapters import GPT2Adapter
from lema._config import LemaConfig, TrainingMode
from lema._full_ft import FullFTManager


def make_gpt2(tmp_path, n_layer=3):
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=n_layer, n_head=2,
                     attn_implementation="eager")
    import transformers
    model = transformers.GPT2LMHeadModel(cfg)
    path = tmp_path / "m.safetensors"
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()}, str(path))
    return cfg, path


def make_manager(tmp_path, cfg_kwargs, hf_cfg, path):
    lema_cfg = LemaConfig(model_name_or_path=str(path), model_type="gpt2",
                          gbi_path=str(path), device="cpu", dtype="float32", **cfg_kwargs)
    assert lema_cfg.training_mode == TrainingMode.SELECTIVE_FULL
    adapter = GPT2Adapter(hf_cfg.to_dict())
    gbi = GlobalBinaryIndex(str(path))
    return FullFTManager(gbi, adapter, lema_cfg)


def test_empty_selection_raises(tmp_path):
    import pytest
    hf_cfg, path = make_gpt2(tmp_path)
    with pytest.raises(ValueError):
        make_manager(tmp_path, {"training_mode": "selective_full",
                                "trainable_modules": ["nope_not_a_module"]}, hf_cfg, path)


def test_module_pattern_selects_across_all_layers(tmp_path):
    hf_cfg, path = make_gpt2(tmp_path)
    mgr = make_manager(tmp_path, {"training_mode": "selective_full",
                                  "trainable_modules": ["c_attn"]}, hf_cfg, path)
    # 3 decoder layers, each c_attn has weight (3*n_embd, n_embd) + bias (3*n_embd,)
    assert mgr.total_selected_params() == 3 * (16 * 3 * 16 + 16 * 3)
    for layer_id in (1, 2, 3):
        names = mgr.selected[layer_id]
        assert all("c_attn" in n for n in names)


def test_layer_range_last_k(tmp_path):
    hf_cfg, path = make_gpt2(tmp_path)
    mgr = make_manager(tmp_path, {"training_mode": "selective_full",
                                  "trainable_layers": ["last:1"]}, hf_cfg, path)
    assert set(mgr.selected.keys()) == {3}  # only decoder layer 3


def test_emb_and_head_selectable(tmp_path):
    hf_cfg, path = make_gpt2(tmp_path)
    mgr = make_manager(tmp_path, {"training_mode": "selective_full",
                                  "trainable_layers": ["emb", "head"]}, hf_cfg, path)
    assert set(mgr.selected.keys()) == {0, hf_cfg.n_layer + 1}


def test_modules_times_layers_intersection(tmp_path):
    hf_cfg, path = make_gpt2(tmp_path)
    mgr = make_manager(tmp_path, {"training_mode": "selective_full",
                                  "trainable_modules": ["c_proj"],
                                  "trainable_layers": ["last:1"]}, hf_cfg, path)
    names = mgr.selected[3]
    assert all("c_proj" in n for n in names)
    # "c_proj" matches both attn.c_proj (16*16) and mlp.c_proj (16*4*16 in GPT2),
    # each with a bias of 16 -> layer 3 only
    assert mgr.total_selected_params() == (16 * 16 + 16) + (16 * 4 * 16 + 16)


def test_default_empty_is_whole_model(tmp_path):
    hf_cfg, path = make_gpt2(tmp_path)
    mgr = make_manager(tmp_path, {"training_mode": "selective_full"}, hf_cfg, path)
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        all_params = sum(f.get_tensor(k).numel() for k in f.keys())
        tied_lm = f.get_tensor("lm_head.weight").numel() if hf_cfg.tie_word_embeddings else 0
    # Whole-model selection covers every unique weight; on tied configs lm_head
    # duplicates the tied wte/embed_tokens weight and is therefore not selected
    # as a separately-trainable tensor.
    assert mgr.total_selected_params() == all_params - tied_lm

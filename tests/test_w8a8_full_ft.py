import math
import os

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from lema import LemaConfig, LemaModel, MemoryStrategy
from lema._config import TrainingMode


def _build(tmp_path, training_mode=TrainingMode.SELECTIVE_FULL, **cfg_kwargs):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="llama",
                    gbi_path=str(model_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0,
                    training_mode=training_mode,
                    trainable_modules=["q_proj"], trainable_layers=["first:1"],
                    grad_accum_backend="ram", output_dir=str(tmp_path / "out"),
                    learning_rate=1e-4, **cfg_kwargs)
    return LemaModel(lc)


def _int8_flat(model, layer_id):
    tr = model.store.transfer
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    return tr.get_vram_flat_buffer(0)


def test_full_ft_apply_to_quantized_linear(tmp_path):
    model = _build(tmp_path, weights_bits=8)
    mgr = model.full_ft_manager
    assert mgr is not None
    layer_id = 1
    flat = _int8_flat(model, layer_id)
    block = model.adapter.construct_layer_module(layer_id, flat, None, mgr)
    qlins = [m for _, m in block.named_modules() if hasattr(m, "weight_int8")]
    assert len(qlins) >= 3
    for mod in qlins:
        assert mod.weight_int8.dtype == torch.int8
        assert mod.scale_w.dtype == torch.float32
    selected = [m for n, m in block.named_modules()
                if hasattr(m, "weight_int8") and n.endswith("q_proj")]
    assert len(selected) == 1
    w_true = mgr.true_weights[(1, "model.layers.0.self_attn.q_proj.weight")]
    q_ref, s_ref = __import__("lema._quant", fromlist=["quantize_tensor"]).quantize_tensor(w_true, 8)
    mod = selected[0]
    w_mod = mod.weight_int8.t().float() * mod.scale_w.view(-1, 1)
    rel = (w_mod - w_true.float()).abs().max() / w_true.float().abs().max()
    assert rel.item() < 1e-1, f"applied int8 weight rel error {rel.item()}"
    assert torch.allclose(mod.scale_w, s_ref.view(-1), atol=1e-6)


def test_full_ft_step_updates_quantized_copy(tmp_path):
    model = _build(tmp_path, weights_bits=8)
    mgr = model.full_ft_manager
    layer_id = 1
    key = (1, "model.layers.0.self_attn.q_proj.weight")
    before = mgr.true_weights[key].clone()
    mgr.accumulators[key].normal_(0, 1)
    mgr.step_layer(layer_id)
    after = mgr.true_weights[key]
    assert not torch.equal(before, after)
    flat = _int8_flat(model, layer_id)
    block = model.adapter.construct_layer_module(layer_id, flat, None, mgr)
    mod = dict(block.named_modules())["self_attn.q_proj"]
    w_mod = mod.weight_int8.t().float() * mod.scale_w.view(-1, 1)
    rel = (w_mod - after.float()).abs().max() / after.float().abs().max()
    assert rel.item() < 1e-1


def test_lora_with_int8_flat_raises(tmp_path):
    model = _build(tmp_path, training_mode=TrainingMode.LORA, weights_bits=8)
    model.initialize_lora()
    layer_id = 1
    flat = _int8_flat(model, layer_id)
    try:
        model.adapter.construct_layer_module(layer_id, flat, model.lora_manager, None)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "int8 flat + lora_manager must fail loudly, not silently drop adapters"


def test_full_ft_train_step_smoke_under_w8a8_buffers(tmp_path):
    torch.manual_seed(0)
    model = _build(tmp_path, weights_bits=8)
    from lema import LemaTrainer
    ids = torch.randint(0, 50, (1, 16))
    trainer = LemaTrainer(config=model.config, model_adapter=model.adapter, gbi=model.gbi,
                          lora_manager=model.lora_manager, store=model.store,
                          full_ft_manager=model.full_ft_manager)
    logits, loss = trainer.train_step(ids, labels=ids.clone())
    assert math.isfinite(loss) and loss > 0

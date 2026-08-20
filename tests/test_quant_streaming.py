import os
import math
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file
from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._config import TrainingMode
from lema._tensorstore import KVChunkStore, chunked_attention


def test_quant_bits_validation_accepts_off_and_valid():
    LemaConfig(model_name_or_path="x", weights_bits=None, opt_state_bits=0,
               grad_acc_bits=8, kv_bits=8)
    LemaConfig(model_name_or_path="x", weights_bits=4)
    LemaConfig(model_name_or_path="x", weights_bits=8)


def test_quant_bits_validation_rejects_invalid():
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", weights_bits=5)
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", opt_state_bits=4)  # 4 only for weights
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", kv_bits=4)  # int8 only for KV
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", grad_acc_bits=16)


def test_quant_bits_serialize_roundtrip(tmp_path):
    c = LemaConfig(model_name_or_path="x", weights_bits=8, kv_bits=8)
    c.save_pretrained(str(tmp_path))
    c2 = LemaConfig.from_pretrained(str(tmp_path))
    assert c2.weights_bits == 8 and c2.kv_bits == 8


def _build_llama(tmp_path, **cfg_kwargs):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="llama",
                    gbi_path=str(model_path), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, **cfg_kwargs)
    return LemaModel(lc)


def test_quantized_weight_stream_loads_and_matches(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    assert tr.quant_bits == 8
    layer_id = 1
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = model.adapter.construct_layer_module(layer_id, flat, None)

    # compare against fp16 baseline weights from disk
    q = model.gbi.load_tensors(["model.layers.0.self_attn.q_proj.weight"], device="cpu")
    w_ref = q["model.layers.0.self_attn.q_proj.weight"]
    w_lema = block.self_attn.q_proj.weight.detach()
    rel = (w_lema.float() - w_ref.float()).abs().max() / w_ref.float().abs().max()
    assert rel.item() < 1e-2, f"quantized weight rel error {rel.item()}"


def test_quantized_forward_close_to_fp16(tmp_path):
    model_q = _build_llama(tmp_path, weights_bits=8)
    model_fp = _build_llama(tmp_path, weights_bits=None)
    hidden = torch.randn(1, 16, model_q.adapter.hidden_size)
    layer_id = 1
    outs = []
    for model in (model_q, model_fp):
        tr = model.store.transfer
        tr.prefetch_to_ram(layer_id, slot=0)
        tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        block = model.adapter.construct_layer_module(layer_id, flat, None)
        block.eval()
        outs.append(model.adapter.forward_layer(block, hidden))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    assert diff < 0.1, f"quantized vs fp16 forward max diff {diff}"


def test_ram_slot_repack_does_not_corrupt_vram_dequant(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    layer_a, layer_b = 1, 2
    tr.prefetch_to_ram(layer_a, slot=0)
    tr.async_transfer_to_vram(layer_a, vram_slot=0, ram_slot=0)
    tr.prefetch_to_ram(layer_b, slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = model.adapter.construct_layer_module(layer_a, flat, None)
    q = model.gbi.load_tensors(["model.layers.0.self_attn.q_proj.weight"], device="cpu")
    w_ref = q["model.layers.0.self_attn.q_proj.weight"]
    rel = (block.self_attn.q_proj.weight.float() - w_ref.float()).abs().max() / w_ref.float().abs().max()
    assert rel.item() < 1e-2, f"dequant corrupted by slot repack: rel error {rel.item()}"


def test_cross_slot_vram_transfer_dequant_matches(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    layer_id = 1
    tr.prefetch_to_ram(layer_id, slot=1)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=1)
    flat = tr.get_vram_flat_buffer(0)
    block = model.adapter.construct_layer_module(layer_id, flat, None)
    q = model.gbi.load_tensors(["model.layers.0.self_attn.q_proj.weight"], device="cpu")
    w_ref = q["model.layers.0.self_attn.q_proj.weight"]
    rel = (block.self_attn.q_proj.weight.float() - w_ref.float()).abs().max() / w_ref.float().abs().max()
    assert rel.item() < 1e-2, f"cross-slot dequant rel error {rel.item()}"


def test_quantized_train_step_with_odd_prefetch_distance(tmp_path):
    torch.manual_seed(0)
    model_q = _build_llama(tmp_path, weights_bits=8, prefetch_distance=3)
    model_fp = _build_llama(tmp_path, weights_bits=None, prefetch_distance=3)
    from lema import LemaTrainer
    ids = torch.randint(0, 50, (1, 16))
    losses = []
    for model in (model_q, model_fp):
        trainer = LemaTrainer(config=model.config, model_adapter=model.adapter, gbi=model.gbi,
                              lora_manager=model.lora_manager, store=model.store)
        _, loss = trainer.train_step(ids, labels=ids.clone())
        losses.append(loss)
    assert math.isfinite(losses[0]) and losses[0] > 0
    assert abs(losses[0] - losses[1]) < 0.2, f"quantized vs fp16 dist=3 train loss diverged: {losses[0]} vs {losses[1]}"


def _build_ft_model(tmp_path, **cfg_kwargs):
    torch.manual_seed(0)
    cfg_kwargs.setdefault("grad_accum_backend", "ram")
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(
        model_name_or_path=str(model_dir), model_type="llama", gbi_path=str(model_dir / "model.safetensors"),
        device="cpu", strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0,
        training_mode=TrainingMode.SELECTIVE_FULL,
        trainable_modules=["q_proj"], trainable_layers=["last:1"],
        output_dir=str(tmp_path / "out"),
        learning_rate=1e-4, **cfg_kwargs,
    )
    return LemaModel(lc)


def test_quantized_opt_state_memory_and_step(tmp_path):
    model = _build_ft_model(tmp_path, opt_state_bits=8)
    mgr = model.full_ft_manager
    key = next(iter(mgr.opt_states))
    st = mgr.opt_states[key]
    q_int, scale = st["exp_avg"]
    assert q_int.dtype == torch.int8
    mgr.accumulators[key].normal_(0, 1)
    mgr.step_layer(key[0])
    q_int2, scale2 = mgr.opt_states[key]["exp_avg"]
    assert q_int2.dtype == torch.int8
    assert scale2.dtype == torch.float32
    assert (scale2 > 0).all() and (q_int2 != 0).any()


def test_quantized_opt_state_tracks_fp32_reference(tmp_path):
    torch.manual_seed(0)
    q_model = _build_ft_model(tmp_path, opt_state_bits=8)
    fp_model = _build_ft_model(tmp_path, opt_state_bits=None)
    q_mgr, fp_mgr = q_model.full_ft_manager, fp_model.full_ft_manager
    for _ in range(50):
        for layer_id in fp_mgr.selected_layer_keys:
            for key in fp_mgr.opt_states:
                g = torch.randn_like(fp_mgr.accumulators[key]) * 0.01
                fp_mgr.accumulators[key].copy_(g)
                q_mgr.accumulators[key].copy_(g)
            fp_mgr.step_layer(layer_id)
            q_mgr.step_layer(layer_id)
    for key in fp_mgr.true_weights:
        w_fp = fp_mgr.true_weights[key].float()
        w_q = q_mgr.true_weights[key].float()
        rel = (w_q - w_fp).abs().max() / (w_fp.abs().max() + 1e-9)
        assert rel.item() < 5e-2, f"quantized full-FT weights diverge after 50 steps: {rel.item()}"


def test_quantized_grad_acc_disk_backend(tmp_path):
    model = _build_ft_model(tmp_path, grad_acc_bits=8, grad_accum_backend="disk")
    mgr = model.full_ft_manager
    assert mgr.accumulator_backend == "disk"
    key = next(iter(mgr.accumulators))
    q_int, scale = mgr.accumulators[key]
    assert q_int.dtype == torch.int8
    assert scale.dtype == torch.float32


def test_quantized_kv_chunk_roundtrip():
    torch.manual_seed(0)
    store_q = KVChunkStore(kv_chunk_size=8, bits=8)
    store_fp = KVChunkStore(kv_chunk_size=8, bits=None)
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    store_q.stash(1, 0, k, v)
    store_fp.stash(1, 0, k, v)
    kq, vq = store_q.load(1, 0)
    kf, vf = store_fp.load(1, 0)
    assert kq.shape == k.shape and vq.shape == v.shape
    assert (kq - kf).abs().max().item() < 0.05
    assert (vq - vf).abs().max().item() < 0.05


def test_quantized_kv_attention_close_to_plain(tmp_path):
    torch.manual_seed(1)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 32, 16)
    v = torch.randn(1, 4, 32, 16)
    sq = KVChunkStore(kv_chunk_size=8, bits=8)
    sf = KVChunkStore(kv_chunk_size=8, bits=None)
    for c in range(4):
        sq.stash(1, c, k[:, :, c*8:(c+1)*8], v[:, :, c*8:(c+1)*8])
        sf.stash(1, c, k[:, :, c*8:(c+1)*8], v[:, :, c*8:(c+1)*8])
    out_q = chunked_attention(q, [sq.load(1, c) for c in range(4)])
    out_fp = chunked_attention(q, [sf.load(1, c) for c in range(4)])
    diff = (out_q - out_fp).abs().max().item()
    assert diff < 0.05, f"quantized KV attention max diff {diff}"


def test_quantized_kv_disk_mmap(tmp_path):
    store = KVChunkStore(kv_chunk_size=8, bits=8, disk_dir=str(tmp_path), max_ram_gb=0)
    k = torch.randn(1, 4, 8, 16)
    store.stash(1, 0, k, k)
    kq, vq = store.load(1, 0)
    assert kq.shape == k.shape and vq.shape == k.shape
    assert (kq - k).abs().max().item() < 0.05
    assert (vq - k).abs().max().item() < 0.05
    import glob
    files = sorted(glob.glob(str(tmp_path / "kv_1_0_*")))
    assert len(files) == 4


def test_quantized_kv_append_grows_chunk():
    torch.manual_seed(2)
    tokens = [(torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16)) for _ in range(6)]
    store_q = KVChunkStore(kv_chunk_size=4, bits=8)
    store_fp = KVChunkStore(kv_chunk_size=4, bits=None)
    for store in (store_q, store_fp):
        for k, v in tokens[:3]:
            store.append(1, k, v)
        assert store.current_size(1) == 3
        assert store.num_chunks(1) == 1
        for k, v in tokens[3:]:
            store.append(1, k, v)
        assert store.num_chunks(1) == 2
    kq, vq = store_q.load(1, 0)
    kf, vf = store_fp.load(1, 0)
    assert kq.shape == kf.shape == (2, 4, 4, 16)
    assert (kq - kf).abs().max().item() < 0.05
    assert (vq - vf).abs().max().item() < 0.05


def test_quantized_kv_disk_append_grows(tmp_path):
    torch.manual_seed(3)
    tokens = [(torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16)) for _ in range(5)]
    store = KVChunkStore(kv_chunk_size=4, bits=8, disk_dir=str(tmp_path), max_ram_gb=0)
    for k, v in tokens[:3]:
        store.append(1, k, v)
    kq, _ = store.load(1, 0)
    assert kq.shape == (2, 4, 3, 16)
    assert store.num_chunks(1) == 1
    store.append(1, tokens[3][0], tokens[3][1])
    assert store.num_chunks(1) == 1
    kq, _ = store.load(1, 0)
    assert kq.shape == (2, 4, 4, 16)
    store.append(1, tokens[4][0], tokens[4][1])
    assert store.num_chunks(1) == 2
    kq, _ = store.load(1, 0)
    assert kq.shape == (2, 4, 4, 16)
    kq2, _ = store.load(1, 1)
    assert kq2.shape == (2, 4, 1, 16)

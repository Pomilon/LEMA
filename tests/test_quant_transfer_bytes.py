"""Quantized streaming must pack raw int8 bytes into RAM staging and transfer
exactly the quantized layer byte count (not a dtype-sized full buffer)."""
import os
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors import safe_open
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._quant_backend import quantize_tensor_with_backend


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


def _quantize_checkpoint_to(src: str, out_dir: str):
    from lema._quant_backend import quantize_tensor_with_backend
    tensors, scales = {}, {}
    with safe_open(src, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            t = handle.get_tensor(key)
            if t.dtype in (torch.int8, torch.uint8) or t.ndim == 0:
                tensors[key] = t.contiguous()
                continue
            q, sc = quantize_tensor_with_backend(t, 8, backend="custom")
            tensors[key] = q.contiguous()
            scales[f"{key}.scale"] = sc.contiguous().float()
    os.makedirs(out_dir, exist_ok=True)
    save_file({**tensors, **scales}, os.path.join(out_dir, "model.safetensors"))


def test_quant_ram_staging_holds_raw_int8_bytes(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    assert tr.quant_bits == 8
    assert tr.ram_buffers[1000].dtype == torch.uint8, \
        f"RAM staging should hold raw bytes, got {tr.ram_buffers[1000].dtype}"
    assert tr.ram_buffers[1000].element_size() == 1
    assert tr.vram_flat_buffers[0].dtype == torch.int8  # native W8A8 VRAM
    assert tr.itemsize == 1, "footprint accounting must use quantized itemsize"


def test_quant_transfer_moves_exact_payload_bytes(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    layer_id = 0  # embed layer: smaller than max slot -> tail must stay untouched
    expected_n = tr._layer_q_numel(layer_id)
    assert 0 < expected_n < tr.max_params

    vram = tr.vram_flat_buffers[0]
    vram.fill_(-123)
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)

    assert tr._ram_packed_bytes.get(1000) == expected_n, \
        "packer must record the quantized payload size"
    assert (tr.vram_flat_buffers[0][expected_n:] == -123).all(), \
        "transfer moved more bytes than the quantized payload (full-buffer copy)"


def test_fp16_transfer_unaffected(tmp_path):
    model = _build_llama(tmp_path)  # no weights_bits
    tr = model.store.transfer
    assert tr.ram_buffers[1000].dtype == tr.dtype
    assert tr.vram_flat_buffers[0].dtype == tr.dtype
    layer_id = 0
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    w = model.gbi.load_tensors(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    got = flat[: w.numel()].view(w.shape)
    assert torch.equal(got, w.to(flat.dtype)), "fp16 path must remain a straight copy"


def test_non_native_quant_vram_is_byte_buffer(tmp_path, monkeypatch):
    from lema import _w8a8
    monkeypatch.setattr(_w8a8, "HAS_NATIVE", False)
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    assert tr.vram_flat_buffers[0].dtype == torch.uint8
    layer_id = 0
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.vram_flat_buffers[0].fill_(200)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    n = tr._layer_q_numel(layer_id)
    assert (tr.vram_flat_buffers[0][n:] == 200).all(), \
        "non-native quant transfer must move only packed bytes"
    flat = tr.get_vram_flat_buffer(0)
    assert flat.dtype == tr.dtype  # dequantized view returned for consumers
    q = model.gbi.load_tensors(["model.embed_tokens.weight"], device="cpu")
    w_ref = q["model.embed_tokens.weight"].float()
    rel = (flat.view(w_ref.shape).float() - w_ref).abs().max() / w_ref.abs().max()
    assert rel.item() < 2e-2, f"dequantized embed rel error {rel.item()}"


@pytest.mark.parametrize("bits", [8, 4])
def test_prequant_checkpoint_stream_and_forward(tmp_path, bits):
    import json
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

    preq_dir = tmp_path / "preq"
    tensors, scales, logical = {}, {}, {}
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            t = handle.get_tensor(key)
            q, sc = quantize_tensor_with_backend(t, bits, backend="custom")
            tensors[key] = q.contiguous()
            scales[f"{key}.scale"] = sc.contiguous().float()
            logical[key] = list(t.shape)
    preq_dir.mkdir(exist_ok=True)
    cfg.save_pretrained(str(preq_dir))
    meta = {"lema_logical_shapes": json.dumps(logical)}
    save_file({**tensors, **scales}, str(preq_dir / "model.safetensors"), metadata=meta)

    lc = LemaConfig(model_name_or_path=str(preq_dir), model_type="llama",
                    gbi_path=str(preq_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, weights_bits=bits)
    model = LemaModel(lc)
    tr = model.store.transfer
    assert tr.ram_buffers[1000].dtype == torch.uint8, "pre-quant RAM staging must be byte-typed"

    layer_id = 1
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    n = tr._layer_q_numel(layer_id)
    assert tr._ram_packed_bytes.get(1000) == n, "pre-quant transfer must move only quantized payload"

    flat = tr.get_vram_flat_buffer(0, allow_quantized=True)
    block = model.adapter.construct_layer_module(layer_id, flat, None)
    block.eval()
    hidden = torch.randn(1, 8, model.adapter.hidden_size)
    with torch.no_grad():
        out_q = model.adapter.forward_layer(block, hidden)

    model_fp = _build_llama(tmp_path)
    tr_fp = model_fp.store.transfer
    tr_fp.prefetch_to_ram(layer_id, slot=0)
    tr_fp.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat_fp = tr_fp.get_vram_flat_buffer(0)
    block_fp = model_fp.adapter.construct_layer_module(layer_id, flat_fp, None)
    block_fp.eval()
    with torch.no_grad():
        out_fp = model_fp.adapter.forward_layer(block_fp, hidden)

    rel = (out_q.float() - out_fp.float()).abs().max() / out_fp.float().abs().max()
    tol = 5e-2 if bits == 8 else 0.12
    assert rel.item() < tol, f"pre-quant int{bits} forward rel error {rel.item()}"


def quantize8(t: torch.Tensor):
    from lema._quant_backend import quantize_tensor_with_backend
    return quantize_tensor_with_backend(t, 8, backend="custom")


def test_layer_transfer_bytes_helper(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    n0 = tr.layer_transfer_bytes(0)
    n1 = tr.layer_transfer_bytes(1)
    assert 0 < n0 < tr.max_params, "layer 0 payload must be smaller than max slot"
    assert 0 < n1 <= tr.max_params
    assert n0 == tr._layer_q_numel(0)


def test_get_layer_scale_returns_device_scale(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    tr.prefetch_to_ram(1, slot=0)
    s = tr.get_layer_scale(1, 0)
    assert s is not None and s.dtype == torch.float32
    assert str(s.device).startswith(tr.device.split(":")[0])


def _save_prequant(model_dir, out_dir, bits, keep_norms_float=True):
    """Pre-quant ckpt like the Kaggle builder: ndim==2 -> int8+scale, norms kept float."""
    import json
    tensors, logical = {}, {}
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            t = handle.get_tensor(key)
            logical[key] = list(t.shape)
            if t.ndim == 2:
                q, sc = quantize_tensor_with_backend(t, bits, backend="custom")
                tensors[key] = q.contiguous()
                tensors[f"{key}.scale"] = sc.contiguous().float()
            else:
                tensors[key] = t.contiguous()
    out = model_dir / out_dir
    out.mkdir(exist_ok=True)
    import shutil
    for f in model_dir.glob("*.json"):
        shutil.copy(f, out / f.name)
    meta = {"lema_logical_shapes": json.dumps(logical)} if bits == 4 else None
    save_file(tensors, str(out / "model.safetensors"), metadata=meta)
    return out


def test_prequant_engine_dtype_matches_model_dtype(tmp_path):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg).to(torch.bfloat16)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))

    preq_dir = _save_prequant(model_dir, "preq", 8)
    lc = LemaConfig(model_name_or_path=str(preq_dir), model_type="llama",
                    gbi_path=str(preq_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, weights_bits=8)
    model = LemaModel(lc)
    # first float non-.scale tensor is a bf16 norm; engine must adopt bf16, not config-default fp16
    assert model.store.transfer.dtype == torch.bfloat16, \
        f"engine dtype {model.store.transfer.dtype} must match model dtype"


def test_prequant_fullft_true_weights_are_dequantized(tmp_path):
    import transformers
    from lema._full_ft import FullFTManager
    from lema.adapters import LlamaAdapter
    from lema._gbi import GlobalBinaryIndex

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

    preq_dir = _save_prequant(model_dir, "preq", 8)
    preq_path = preq_dir / "model.safetensors"

    lema_cfg = LemaConfig(model_name_or_path=str(preq_dir), model_type="llama",
                          gbi_path=str(preq_path), device="cpu", dtype="float32",
                          training_mode="selective_full", trainable_layers=["1"])
    gbi = GlobalBinaryIndex(str(preq_path))
    adapter = LlamaAdapter(cfg.to_dict())
    mgr = FullFTManager(gbi, adapter, lema_cfg)

    # original bf16-free fp32 reference weights
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        ref = {k: handle.get_tensor(k).float() for k in handle.keys()}
    for (lid, name), w in mgr.true_weights.items():
        q, sc = quantize_tensor_with_backend(ref[name], 8, backend="custom")
        w_deq = (q.float().view(sc.shape[0], -1) * sc.view(-1, 1)).reshape(ref[name].shape)
        rel = (w.float() - w_deq).abs().max() / w_deq.abs().max()
        assert rel.item() < 1e-2, \
            f"true weight {name} must be dequantized from pre-quant int8 (rel err {rel.item():.3f})"


def test_prequant_bf16_fullft_train_step_no_dtype_crash(tmp_path):
    """Mirrors the v23 Kaggle crash: pre-quant ckpt + bf16 model + selective full-FT."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg).to(torch.bfloat16)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    preq_dir = _save_prequant(model_dir, "preq", 8)

    lc = LemaConfig(model_name_or_path=str(preq_dir), model_type="llama",
                    gbi_path=str(preq_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, weights_bits=8,
                    training_mode="selective_full", trainable_layers=["1"])
    model = LemaModel(lc)
    assert model.store.transfer.dtype == torch.bfloat16
    losses = []
    for _ in range(2):
        ids = torch.randint(0, 100, (1, 16))
        _, loss = model.get_trainer().train_step(ids, labels=ids.clone())
        losses.append(float(loss))
    assert all(l == l and l < 1e3 for l in losses), f"losses diverged: {losses}"
    # second step must differ: guards against updates never landing in the forward
    assert losses[1] != losses[0], f"training did not advance: {losses}"

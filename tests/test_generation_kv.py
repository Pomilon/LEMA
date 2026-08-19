import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy


def _build(tmp_path):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=100, n_positions=64, n_embd=32, n_layer=2, n_head=2,
        attn_implementation="eager",
    )
    model_hf = GPT2LMHeadModel(config)
    state_dict = {k: v.clone().detach() for k, v in model_hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(state_dict, str(model_path))
    config.save_pretrained(str(model_dir))
    lema_config = LemaConfig(
        model_name_or_path=str(model_dir), model_type="gpt2", gbi_path=str(model_path),
        device="cpu", strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, kv_chunk_size=4,
    )
    return LemaModel(lema_config)


def test_generate_kv_matches_old_greedy(tmp_path):
    model = _build(tmp_path)

    class Tok:
        def __init__(self):
            self.eos_token_id = 2
        def __call__(self, prompt, return_tensors="pt"):
            ids = torch.tensor([[1, 3, 5, 7]])
            class _Wrap(dict):
                def to(self, device):
                    self["input_ids"] = self["input_ids"].to(device)
                    return self
            return _Wrap(input_ids=ids)
        def decode(self, ids, skip_special_tokens=True):
            return ",".join(str(int(x)) for x in ids.tolist())

    tok = Tok()
    torch.manual_seed(1)
    out_old = model.generate("x", tok, max_new_tokens=8, do_sample=False)
    torch.manual_seed(1)
    out_kv = model.generate_kv("x", tok, max_new_tokens=8, do_sample=False, kv_chunk_size=4)
    assert out_old == out_kv, f"old={out_old} kv={out_kv}"

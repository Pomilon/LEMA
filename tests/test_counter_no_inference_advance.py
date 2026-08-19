import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._config import TrainingMode


def _build(tmp_path, mode):
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=2,
                     attn_implementation="eager")
    hf = GPT2LMHeadModel(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))
    kwargs = dict(model_name_or_path=str(model_dir), model_type="gpt2", gbi_path=str(model_path),
                  device="cpu", strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0,
                  gradient_accumulation_steps=2)
    if mode == "full_ft":
        kwargs.update(training_mode=TrainingMode.SELECTIVE_FULL,
                      trainable_layers=["last:1"], trainable_modules=["c_attn"])
    return LemaModel(LemaConfig(**kwargs))


def test_generate_does_not_advance_accumulation_counter(tmp_path):
    model = _build(tmp_path, "full_ft")
    trainer = model.get_trainer(None)
    mgr = model.full_ft_manager
    before = mgr.accumulation_step

    class Tok:
        eos_token_id = 2
        def __call__(self, prompt, return_tensors="pt"):
            class _W(dict):
                def to(self, device):
                    self["input_ids"] = self["input_ids"].to(device)
                    return self
            return _W(input_ids=torch.tensor([[1, 3, 5, 7]]))
        def decode(self, ids, skip_special_tokens=True):
            return str(ids)

    model.generate("x", Tok(), max_new_tokens=3, do_sample=False)
    assert mgr.accumulation_step == before, "generate advanced the accumulation counter"


def test_evaluate_does_not_advance_accumulation_counter(tmp_path):
    model = _build(tmp_path, "full_ft")
    trainer = model.get_trainer(None)
    mgr = model.full_ft_manager
    before = mgr.accumulation_step

    class DS:
        def __len__(self):
            return 1
        def __iter__(self):
            yield torch.randint(0, 100, (1, 8))

    trainer.evaluate(DS())
    assert mgr.accumulation_step == before, "evaluate advanced the accumulation counter"


def test_training_advances_accumulation_counter(tmp_path):
    model = _build(tmp_path, "full_ft")
    trainer = model.get_trainer(None)
    mgr = model.full_ft_manager
    ids = torch.randint(0, 100, (1, 8))
    with torch.enable_grad():
        trainer.train_step(ids, labels=ids.clone())
    assert mgr.accumulation_step == 1, "training step should advance the counter once"
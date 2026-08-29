import torch
import tempfile
import os
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM
from lema import LemaConfig, LemaModel, MemoryStrategy


def _tiny_llama(tmp_path):
    hf_cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    hf = LlamaForCausalLM(hf_cfg)
    sd = {k: v.clone() for k, v in hf.state_dict().items()}
    del hf
    path = os.path.join(tmp_path, "model.safetensors")
    save_file(sd, path)
    hf_cfg.save_pretrained(tmp_path)
    return path


def test_each_backend_trains(tmp_path):
    path = _tiny_llama(str(tmp_path))
    for backend in ["custom", "torchao", "quanto", "bitsandbytes", "auto"]:
        cfg = LemaConfig(model_name_or_path=str(tmp_path), model_type="llama", gbi_path=path, device="cpu", strategy=MemoryStrategy.STREAMING, training_mode="selective_full", trainable_modules=["q_proj"], trainable_layers=["last:1"], weights_bits=8, quant_backend=backend, dtype="float32")
        model = LemaModel(cfg)
        trainer = model.get_trainer()
        ids = torch.randint(0, 100, (1, 16))
        _, loss = trainer.train_step(ids, labels=ids)
        assert torch.isfinite(torch.tensor(loss))
        model.close()


def test_transfer_bytes_reduced_per_backend(tmp_path):
    path = _tiny_llama(str(tmp_path))
    for backend in ["custom", "torchao", "quanto"]:
        cfg = LemaConfig(model_name_or_path=str(tmp_path), model_type="llama", gbi_path=path, device="cpu", strategy=MemoryStrategy.STREAMING, weights_bits=8, quant_backend=backend, dtype="float32")
        model = LemaModel(cfg)
        tr = model.store.transfer
        assert tr._layer_q_numel(2) < tr.max_params * 4
        model.close()

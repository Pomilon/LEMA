import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file
from lema import LemaConfig, LemaModel, LemaTrainer, MemoryStrategy
from lema._config import TrainingMode


def _build(tmp_path, prefetch_distance):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "m"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="llama",
                    gbi_path=str(model_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0,
                    prefetch_distance=prefetch_distance, training_mode=TrainingMode.LORA,
                    output_dir=str(tmp_path / "out"), learning_rate=1e-3)
    model = LemaModel(lc)
    trainer = LemaTrainer(
        config=model.config,
        model_adapter=model.adapter,
        gbi=model.gbi,
        lora_manager=model.lora_manager,
        store=model.store,
    )
    return trainer, lc


def test_train_step_weights_match_at_prefetch_distance_3(tmp_path):
    """With 3 layers and prefetch_distance=3, layer 0's forward must use layer 0's weights."""
    trainer, _ = _build(tmp_path, prefetch_distance=3)
    input_ids = torch.randint(0, 50, (1, 16))
    labels = input_ids.clone()
    _, loss_a = trainer.train_step(input_ids, labels)
    trainer2, _ = _build(tmp_path, prefetch_distance=2)
    _, loss_b = trainer2.train_step(input_ids, labels)
    assert abs(loss_a - loss_b) < 1e-3, f"dist=3 loss {loss_a} diverges from dist=2 {loss_b}"
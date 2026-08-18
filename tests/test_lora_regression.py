import torch
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file
from lema import LemaConfig, LemaModel, MemoryStrategy


def test_lora_path_still_works_after_full_ft_refactor(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"

    hf_cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                        attn_implementation="eager", tie_word_embeddings=False)
    model = GPT2LMHeadModel(hf_cfg)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(model_path))
    hf_cfg.save_pretrained(str(model_dir))

    lema_cfg = LemaConfig(
        model_name_or_path=str(model_dir),
        model_type="gpt2",
        gbi_path=str(model_path),
        device="cpu",
        dtype="float32",
        strategy=MemoryStrategy.STREAMING,
        learning_rate=0.1,
        lora_rank=2,
        lora_target_modules=["c_attn"],
        save_steps=2,
        max_ram_gb=64,
        output_dir=str(tmp_path / "checkpoints"),
    )

    model = LemaModel(lema_cfg)
    assert model.full_ft_manager is None, "default mode must remain LoRA after full-FT refactor"
    assert model.lora_manager is not None
    model.initialize_lora()

    trainable = list(model.get_trainable_parameters())
    assert len(trainable) > 0
    assert all(p.requires_grad for p in trainable)

    optimizer = optim.SGD(model.get_trainable_parameters(), lr=lema_cfg.learning_rate)
    trainer = model.get_trainer(optimizer)
    assert trainer.full_ft_manager is None
    assert trainer.lora_manager is not None

    input_ids = torch.randint(0, 64, (1, 10))
    initial_params = [p.clone() for p in model.get_trainable_parameters()]
    for _ in range(4):
        _, loss = trainer.train_step(input_ids, labels=input_ids)
        assert loss is not None and loss > 0

    changed = any(not torch.allclose(a, b) for a, b in zip(initial_params, model.get_trainable_parameters()))
    assert changed, "LoRA parameters did not change"

    checkpoint = tmp_path / "checkpoints" / "checkpoint-4"
    assert checkpoint.exists()
    assert (checkpoint / "adapter_model.bin").exists(), "LoRA checkpoint must contain adapter weights"

    save_dir = tmp_path / "saved_model"
    model.save_pretrained(str(save_dir))
    assert (save_dir / "adapter_model.bin").exists()

    loaded = LemaModel.from_pretrained(str(save_dir))
    assert loaded.full_ft_manager is None
    for p_final, p_loaded in zip(model.get_trainable_parameters(), loaded.get_trainable_parameters()):
        assert torch.allclose(p_final, p_loaded), "LoRA weights did not round-trip"

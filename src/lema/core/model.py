import torch
import os
import math
import gc
import psutil
import time
from typing import Optional, Dict, Any, Union, List
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import snapshot_download

from ..config import LemaConfig, MemoryStrategy
from ..models import get_adapter
from .gbi import GlobalBinaryIndex
from .lora import LoRAManager
from .memory import TripleBufferManager
from ..utils.logger import logger
from ..utils.conversion import convert_to_monolith
from ..engine.trainer import LemaTrainer

class LemaModel:
    """
    High-level interface for LEMA Models.
    Wraps all low-level components into a single object and provides 
    unified methods for training and inference.
    """
    def __init__(self, config: Union[LemaConfig, str]):
        if isinstance(config, str) or (hasattr(config, '__fspath__')):
            self.config = LemaConfig.from_pretrained(str(config))
        else:
            self.config = config
        
        # 0. Hardware Safety: Set VRAM fraction if requested
        if self.config.device.startswith("cuda") and self.config.vram_fraction < 1.0:
             torch.cuda.set_per_process_memory_fraction(self.config.vram_fraction)
             logger.info(f"LEMA: Capping PyTorch VRAM usage to {self.config.vram_fraction*100:.1f}%")

        # 1. Initialize GBI (Multi-file Support)
        # If gbi_path is just the default "model.safetensors" and it doesn't exist,
        # we check if model_name_or_path provides everything we need.
        source_path = self.config.model_name_or_path
        
        if not os.path.exists(source_path):
            try:
                logger.info(f"LEMA: Resolving '{source_path}' via Hugging Face Hub...")
                source_path = snapshot_download(source_path, allow_patterns=["*.safetensors", "*.json"])
            except Exception as e:
                logger.error(f"LEMA: Failed to resolve source model '{source_path}': {e}")
                raise FileNotFoundError(f"Source model '{source_path}' not found. Check LemaConfig.")

        # Check if we still need a monolith or if GBI can handle the directory
        if not os.path.exists(self.config.gbi_path):
             # If source_path is a directory with safetensors, GBI handles it!
             # We only need 'conversion' if we explicitly want a single file or if we have .bin files.
             if os.path.isdir(source_path) and any(f.endswith(".safetensors") for f in os.listdir(source_path)):
                 logger.info(f"LEMA: Using multi-file SafeTensors from {source_path}")
                 self.gbi_path = source_path # Redirect GBI to the directory
             elif source_path.endswith(".safetensors"):
                 self.gbi_path = source_path
             else:
                 # Real conversion needed (e.g. from .bin or other formats)
                 logger.info(f"LEMA: Converting model from {source_path} to {self.config.gbi_path}...")
                 convert_to_monolith(source_path, self.config.gbi_path)
                 self.gbi_path = self.config.gbi_path
        else:
            self.gbi_path = self.config.gbi_path
        
        self.gbi = GlobalBinaryIndex(self.gbi_path)
        
        # 2. Get HF config for the adapter
        try:
            hf_config_obj = AutoConfig.from_pretrained(self.config.model_name_or_path)
            hf_config_dict = hf_config_obj.to_dict()
        except Exception as e:
            logger.warning(f"LEMA: AutoConfig failed to load from {self.config.model_name_or_path} ({e}). Using config from LemaConfig.")
            hf_config_dict = self.config.to_dict()

        # 3. Initialize Adapter
        model_type = self.config.model_type
        if model_type is None:
            # Auto-detect from HF config
            model_type = hf_config_dict.get("model_type", "llama")
            logger.info(f"LEMA: Auto-detected model type: {model_type}")
        
        self.adapter = get_adapter(model_type, hf_config_dict)
        
        # 4. Determine model dtype from GBI sample
        model_dtype = torch.float32
        if self.gbi.get_keys():
            try:
                sample = self.gbi.load_tensors([self.gbi.get_keys()[0]])
                model_dtype = next(iter(sample.values())).dtype
            except: pass

        # 5. Initialize LoRA Manager with matching dtype
        self.lora_manager = LoRAManager({
            "r": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "target_modules": self.config.lora_target_modules
        }, device=self.config.device, dtype=model_dtype)
        
        # 6. Initialize Memory Manager
        self.memory = TripleBufferManager(
            self.gbi, 
            self.adapter, 
            config=self.config
        )


    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Loads a LEMA model and its adapters from a directory."""
        config = LemaConfig.from_pretrained(path, **kwargs)
        model = cls(config)
        
        # Load adapters if they exist
        if os.path.exists(os.path.join(path, "adapter_model.bin")):
            model.lora_manager.load_pretrained(path)
            
        return model

    def save_pretrained(self, save_directory: str):
        """Saves the configuration and LoRA adapters."""
        self.config.save_pretrained(save_directory)
        self.lora_manager.save_pretrained(save_directory)

    def initialize_lora(self):
        """Pre-initializes LoRA adapters and warms the sliding-window pool.
        Creates 3 pool modules (emb, decoder, head) on CPU — first train_step moves to GPU."""
        logger.info("LEMA: Pre-initializing LoRA adapters...")
        layers = self.adapter.get_layer_metadata()
        # Warm pool: one emb, one decoder, one head
        for layer in layers[:3]:  # IDs 0, 1, 2 (emb, decoder, decoder)
            self.adapter.construct_layer_module(layer['id'], None, self.lora_manager)
        # Ensure head module is also warmed
        self.adapter.construct_layer_module(layers[-1]['id'], None, self.lora_manager)

    def get_trainable_parameters(self):
        return self.lora_manager.get_trainable_parameters()

    def get_trainer(self, optimizer: Optional[torch.optim.Optimizer] = None, **kwargs) -> Any:
        """Constructs a trainer with the current or optimized configuration."""
        
        # Pop internal kwargs before passing to LemaTrainer
        auto_optimize = kwargs.pop("auto_optimize", False)
        
        # Auto-optimize if limits are not set
        if self.config.max_ram_gb <= 0 or auto_optimize:
            self.simulate_and_optimize()
            # Re-init memory with optimized config
            self.memory = TripleBufferManager(self.gbi, self.adapter, config=self.config)
            
        return LemaTrainer(
             config=self.config,
             model_adapter=self.adapter,
             gbi=self.gbi,
             lora_manager=self.lora_manager,
             memory_manager=self.memory,
             optimizer=optimizer,
             **kwargs
        )

    def simulate_and_optimize(self):
        """
        Runs a 'Flight Check' dry-run to detect hardware bottlenecks 
        and mathematically find the optimal configuration.
        """
        logger.info("LEMA: Running Sophisticated Dynamic Flight Check...")
        
        # 1. Initialize temporary memory manager for profiling
        # We use a temporary one to avoid side-effects on the main trainer
        temp_mem = TripleBufferManager(self.gbi, self.adapter, config=self.config)
        itemsize = temp_mem.itemsize
        
        # 2. Benchmark Disk -> RAM (Actual Tensors)
        # We sample a few layers to get a realistic I/O profile
        test_layer_id = self.adapter.get_layer_metadata()[0]['id']
        start = time.perf_counter()
        temp_mem._pack_layer_to_ram(test_layer_id, slot=0, is_resident=False)
        disk_time = time.perf_counter() - start
        disk_size_mb = (temp_mem.max_params * itemsize) / (1024**2)
        disk_speed_mb = disk_size_mb / max(disk_time, 1e-6)
        
        # 3. Benchmark RAM -> VRAM (Actual Buffer & Dtype)
        if torch.cuda.is_available():
            start = time.perf_counter()
            # Perform multiple transfers to warm up PCIe and get average
            for _ in range(5):
                temp_mem.async_transfer_to_vram(test_layer_id, vram_slot=0, ram_slot=0)
                temp_mem.get_vram_flat_buffer(0) # Synchronize
            vram_time = (time.perf_counter() - start) / 5
            vram_speed_mb = disk_size_mb / max(vram_time, 1e-6)
        else:
            vram_time = 0.001
            vram_speed_mb = 1000.0
            
        logger.info(f"LEMA: Benchmarked Hardware -> Disk: {disk_speed_mb:.1f} MB/s, PCIe: {vram_speed_mb:.1f} MB/s")

        # 4. Measure Compute Time (Dry Run)
        # We run 2 steps to warm up Cuda graph / kernels
        dummy_optimizer = torch.optim.AdamW(self.get_trainable_parameters(), lr=0)
        temp_trainer = LemaTrainer(
            config=self.config,
            model_adapter=self.adapter,
            gbi=self.gbi,
            lora_manager=self.lora_manager,
            memory_manager=temp_mem,
            optimizer=dummy_optimizer
        )
        
        # Use a real sequence length of 512 (common default)
        dummy_input = torch.randint(0, 100, (1, 512), device=self.config.device)
        
        # Warmup step
        temp_trainer.train_step(dummy_input)
        
        # Profile step
        start = time.perf_counter()
        temp_trainer.train_step(dummy_input)
        step_time = time.perf_counter() - start
        
        # Calculate per-layer compute (roughly step_time / num_layers / 2 for fwd+bwd)
        num_layers = len(self.adapter.get_layer_metadata())
        t_comp_layer = step_time / (num_layers * 2) 
        
        # 5. Sophisticated Optimization Algorithm (Little's Law approximation)
        # We need (prefetch_distance - 1) * T_comp >= T_disk_to_ram
        # because T_ram_to_vram is usually much smaller and overlapping.
        needed_dist = int(math.ceil(disk_time / max(t_comp_layer, 1e-6))) + 1
        
        # Clamp to reasonable values [1, 5] to avoid excessive RAM usage
        self.config.prefetch_distance = max(1, min(needed_dist, 5))
        
        # 6. Final Strategy & Budget Decisions
        total_model_params = sum(l.get('size', 0) for l in self.adapter.get_layer_metadata())
        total_model_gb = (total_model_params * itemsize) / (1024**3)
        
        if self.config.max_ram_gb <= 0:
            total_ram = psutil.virtual_memory().total / (1024**3)
            self.config.max_ram_gb = total_ram * 0.75
            
        if total_model_gb <= self.config.max_ram_gb * 0.9:
            self.config.strategy = MemoryStrategy.RESIDENT
        else:
            self.config.strategy = MemoryStrategy.STREAMING
            
        logger.info(f"LEMA: Opt-Algo result -> T_comp: {t_comp_layer*1000:.1f}ms, T_disk: {disk_time*1000:.1f}ms")
        logger.info(f"LEMA: Final Config -> Prefetch: {self.config.prefetch_distance}, Strategy: {self.config.strategy.value}")
        
        # Cleanup temp manager
        del temp_trainer, temp_mem, dummy_optimizer
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(self, 
                 prompt: str, 
                 tokenizer: Any, 
                 max_new_tokens: int = 50,
                 do_sample: bool = True,
                 temperature: float = 0.7):
        """Simple generation for quick model verification."""
        # Clear per-layer caches (may be stale from training at different seq_len)
        for attr in ['_cache_seq', '_cache_pos', '_cache_mask', '_cache_rope']:
            if hasattr(self.adapter, attr):
                delattr(self.adapter, attr)
        trainer = self.get_trainer(None) 
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_ids = inputs["input_ids"]
        
        logger.info(f"LEMA: Generating sequence (max {max_new_tokens} tokens)...")
        
        for _ in range(max_new_tokens):
            logits, _ = trainer.train_step(input_ids)
            next_token_logits = logits[:, -1, :]
            
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def to(self, device: str):
        self.config.device = device
        self.lora_manager.device = device
        self.memory.device = device
        return self
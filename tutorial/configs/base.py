
import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    batch_size: int = 2
    block_size: int = 1024
    vocab_size: int = 50257
    pad_size: int = 50304  # pad vocab_size to be more efficient
    n_embd: int = 768      # 768, 2048
    n_head: int = 12       # 12, 32
    n_layer: int = 12       # 12, 22
    use_bias: bool = False
    dropout: float = 0.0
    epsilon: float = 1e-5
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = True  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class TrainConfig:
    log_path: str = "log/"
    epoch: int = -1
    total_token_batch_size: int = 1024 * 2  # 524288, 2**19, about 0.5 tokens per batch
    warmup_steps: int = 715
    max_steps: int = 300
    check_every_steps: int = 1
    val_every_steps: int = 250
    save_every_steps: int = 5000
    max_lr: float = 1e-7
    min_lr: float = 0.1 * max_lr
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip_value: float = 1.0
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.bfloat16  # amp precision: torch.bfloat16, torch.float16 (Now we only support bf16)
    seed: int = 42
    wait_every_step: int = 1

@dataclass
class MezoConfig:
    max_zo_random_seed: int = 1000000000
    zo_eps: float = 1e-3
    non_diff: bool = False
    zo_lr: float = 1e-3
    zo_weight_decay: float = 1e-1

@dataclass
class OffloadingConfig:
    offload_to_device: torch.device = "cpu"
    offload_from_device: torch.device = "cuda:0"
    overlap: bool = True    # if you want to make communication-computation overlap, 'True' will be faster.
    offload_every_blocks: int = 1   # how many layers per interval do you want to offload a layer
    empty_cache_every_blocks: int = 1   # frequency of empty cache
    offload_use_amp: bool = True
    offload_amp_dtype: torch.dtype = torch.bfloat16
    offload_upcast_dtype: torch.dtype = torch.float32
    offload_downcast_dtype: torch.dtype = torch.bfloat16
    medium_precision_blocks_on_device: bool = False
    compress_method: str = "naive_quantization"
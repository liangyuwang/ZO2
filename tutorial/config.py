
import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    batch_size: int = 1
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
    dtype: torch.dtype = torch.bfloat16  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "flash_attention"  # "standard_attention", "flash_attention"

@dataclass
class TrainConfig:
    log_path: str = "log/"
    epoch: int = -1
    total_token_batch_size: int = 1024 * 2  # 524288, 2**19, about 0.5 tokens per batch
    warmup_steps: int = 715
    max_steps: int = 50
    check_every_steps: int = 1
    val_every_steps: int = 250
    save_every_steps: int = 5000
    max_lr: float = 6e-3
    min_lr: float = 0.1 * max_lr
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip_value: float = 1.0
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.bfloat16  # amp precision: torch.bfloat16, torch.float16 (Now we only support bf16)
    seed: int = 1337
    wait_every_step: int = 1

@dataclass
class DataConfig:
    path: str = "../../../../dataset/fineweb/fineweb-edu-10BT/"
    num_workers: int = 4
    shuffle: bool = False

@dataclass
class MezoConfig:
    zo_random_seed: int = 42
    zo_eps: float = 1e-3
    non_diff: bool = False
    zo_lr: float = 1e-3
    zo_weight_decay: float = 1e-1


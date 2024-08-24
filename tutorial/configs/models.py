
import torch
from dataclasses import dataclass

from .base import ModelConfig


@dataclass
class GPT2_small(ModelConfig):  # 125 M
    batch_size: int = 1
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = True  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class GPT2_medium(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = True  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class GPT2_large(ModelConfig):  # 
    batch_size: int = 1
    n_embd: int = 1280
    n_head: int = 20
    n_layer: int = 36
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = True  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class GPT2_xl(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 1600
    n_head: int = 25
    n_layer: int = 48
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = True  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_125m(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = True  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_350m(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = True  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_1_3b(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 2048
    n_head: int = 32
    n_layer: int = 24
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_2_7b(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 2560
    n_head: int = 32
    n_layer: int = 32
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_6_7b(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 4096
    n_head: int = 32
    n_layer: int = 32
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_13b(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 5120
    n_head: int = 40
    n_layer: int = 40
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_30b(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 7168
    n_head: int = 56
    n_layer: int = 48
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "flash_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_66b(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 9216
    n_head: int = 72
    n_layer: int = 64
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "flash_attention"  # "standard_attention", "flash_attention"

@dataclass
class OPT_175b(ModelConfig): # 
    batch_size: int = 1
    n_embd: int = 12288
    n_head: int = 96
    n_layer: int = 96
    device: torch.device = "cuda:0"
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "flash_attention"  # "standard_attention", "flash_attention"

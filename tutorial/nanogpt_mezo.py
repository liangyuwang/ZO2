
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from mezo_offload import BaseMezoModel

from .configs import ModelConfig, MezoConfig
from .nanogpt import Block


class GPT2ModelMezo(nn.Module, BaseMezoModel):

    def __init__(self, config: ModelConfig, mezoConfig: MezoConfig):
        super().__init__()
        self.config = config
        self.mezoConfig = mezoConfig
        self.set_mezo_config()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.pad_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.pad_size, bias=False)

        # weight sharing scheme
        if config.share_emb:
            self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        if self.zo_training:
            return self.zo_forward(idx, targets)
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, pad_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    ############## MeZO ##############
    # inspired by https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py

    def set_mezo_config(self):
        self.max_zo_random_seed = self.mezoConfig.max_zo_random_seed
        self.zo_eps = self.mezoConfig.zo_eps
        self.non_diff = self.mezoConfig.non_diff
        self.zo_lr = self.mezoConfig.zo_lr
        self.zo_weight_decay = self.mezoConfig.zo_weight_decay
        self.set_mezo_args()

    @torch.inference_mode
    def zo_forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb1, pos_emb2 = self.zo_dual_forward(self.transformer.wpe, (pos, pos))
        tok_emb1, tok_emb2 = self.zo_dual_forward(self.transformer.wte, (idx, idx))
        x1, x2 = tok_emb1 + pos_emb1, tok_emb2 + pos_emb2
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x1, x2 = self.zo_dual_forward(block, (x1, x2))
        # forward the final layernorm and the classifier
        x1, x2 = self.zo_dual_forward(self.transformer.ln_f, (x1, x2))
        logits1, logits2 = self.zo_dual_forward(self.lm_head, (x1, x2), zero_grad=True)
        loss1 = loss2 = None
        if targets is not None:
            loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets.view(-1))
            loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), targets.view(-1))
            self.zo_final_step(loss1, loss2)
        return (logits1, logits2), (loss1, loss2)
    

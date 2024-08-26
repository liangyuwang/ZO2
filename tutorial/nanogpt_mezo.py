
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import time
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from .configs import ModelConfig, MezoConfig
from .nanogpt import Block


class GPT2ModelMezo(nn.Module):

    def __init__(self, config: ModelConfig, mezoConfig: MezoConfig):
        super().__init__()
        self.config = config
        self.mezoConfig = mezoConfig

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

        self.zo_training = True
        self.projected_grad = 0
        self.grad_accum = False
    
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

    @torch.inference_mode
    def zo_forward(self, idx, targets=None):
        self.set_random_seed()
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
        logits1, logits2 = self.zo_dual_forward(self.lm_head, (x1, x2))
        loss1 = loss2 = None
        if targets is not None:
            loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets.view(-1))
            loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), targets.view(-1))
            self.zo_final_step(loss1, loss2)
        return (logits1, logits2), (loss1, loss2)
    
    @torch.inference_mode
    def zo_dual_forward(self, module:nn.Module, dual_inputs):
        input1, input2 = dual_inputs
        if self.projected_grad != 0:
            self._zo_update(module)
        if not self.grad_accum:
            self._zo_zero_grad()
        cloned_module = self._zo_module_clone(module)
        self._zo_perturb_parameters(cloned_module, scaling_factor=1)
        out1 = cloned_module(input1)
        self._zo_perturb_parameters(cloned_module, scaling_factor=-2)
        out2 = cloned_module(input2)
        del cloned_module
        return out1, out2
    
    @torch.inference_mode
    def zo_final_step(self, loss1, loss2):
        self.projected_grad += (loss1 - loss2) / (2 * self.mezoConfig.zo_eps)
    
    @torch.inference_mode
    def _zo_perturb_parameters(self, module:nn.Module, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        # """
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in module.named_parameters():
            if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.mezoConfig.zo_eps

    @torch.inference_mode
    def _zo_module_clone(self, module:nn.Module):
        cloned_module = deepcopy(module)
        return cloned_module

    @torch.inference_mode
    def _zo_update(self, module:nn.Module):
        torch.manual_seed(self.zo_random_seed)
        for name, param in module.named_parameters():
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if param.requires_grad:
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + self.mezoConfig.zo_weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

    @torch.inference_mode
    def _zo_zero_grad(self):
        self.projected_grad = 0

    @torch.inference_mode
    def _get_learning_rate(self):
        return self.mezoConfig.zo_lr

    @torch.inference_mode
    def set_random_seed(self):
        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(self.mezoConfig.max_zo_random_seed)


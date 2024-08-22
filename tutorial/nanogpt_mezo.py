
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import time
from copy import deepcopy
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# modified from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

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
    dtype: torch.dtype = torch.float32  # model precision
    share_emb: bool = False  # share embedding weight or not. See https://arxiv.org/abs/1706.03762
    attention: str = "standard_attention"  # "standard_attention", "flash_attention"

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

# -----------------------------------------------------------------------------
# Model definition

# Masked Multi-Head Self-Attention

def standard_attention(q, k, v, dropout=True, dropout_p=0.0):
    T = q.size(-3)
    scale = 1.0 / math.sqrt(k.size(-1))
    q = q.transpose(1, 2) # (B, nh, T, hs)
    k = k.transpose(1, 2) # (B, nh, T, hs)
    v = v.transpose(1, 2) # (B, nh, T, hs)
    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(T, T, device=v.device)).view(1, 1, T, T)
    att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = F.dropout(att, p=dropout_p, training=dropout)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y.transpose(1, 2).contiguous() # (B, T, nh, hs)

def flash_attention(q, k, v, dropout=True, dropout_p=0.0):
    scale = 1.0 / math.sqrt(k.size(-1))
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if not dropout: dropout_p=0.0
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True, scale=scale)
    return y.transpose(1, 2)

class CausalSelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attention = config.attention
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.use_bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        # Attention implementation
        if self.attention == "standard_attention":
            y = standard_attention(q, k, v, dropout=self.training, dropout_p=False)
        elif self.attention == "flash_attention":
            y = flash_attention(q, k, v, dropout=self.training, dropout_p=False)
        y = y.view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

# Feed-Forward Neural Network

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.use_bias)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.use_bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer Block

class Block(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        out1 = self.ln_1(x)
        out1 = self.attn(out1)
        x = x + out1
        out2 = self.ln_2(x)
        out2 = self.mlp(out2)
        x = x + out2
        return x

# GPT-2 Model

class GPT2ModelMezo(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

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

    def zo_forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb1, pos_emb2 = self.zo_dual_forward(self.transformer.wpe, (pos, pos))
        tok_emb1, tok_emb2 = self.zo_dual_forward(self.transformer.wte, (idx, idx))
        with torch.inference_mode():
            x1, x2 = tok_emb1 + pos_emb1, tok_emb2 + pos_emb2
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x1, x2 = self.zo_dual_forward(block, (x1, x2))
        # forward the final layernorm and the classifier
        x1, x2 = self.zo_dual_forward(self.transformer.ln_f, (x1, x2))
        logits1, logits2 = self.zo_dual_forward(self.lm_head, (x1, x2))
        loss1 = loss2 = None
        if targets is not None:
            with torch.inference_mode():
                loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets.view(-1))
                loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), targets.view(-1))
                self.zo_final_step(loss1, loss2)
        return (logits1, logits2), (loss1, loss2)
    
    def zo_dual_forward(self, module:nn.Module, dual_inputs):
        input1, input2 = dual_inputs
        if self.projected_grad != 0:
            self._zo_update(module)
        if not self.grad_accum:
            self._zo_zero_grad()
        cloned_module = self._zo_module_clone(module)
        self._zo_perturb_parameters(cloned_module, scaling_factor=1)
        with torch.inference_mode():
            out1 = cloned_module(input1)
        self._zo_perturb_parameters(cloned_module, scaling_factor=-2)
        with torch.inference_mode():
            out2 = cloned_module(input2)
        del cloned_module
        return out1, out2
    
    def zo_final_step(self, loss1, loss2):
        self.projected_grad += (loss1 - loss2) / (2 * mezoConfig.zo_eps)
    
    def _zo_perturb_parameters(self, module:nn.Module, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        # """
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else mezoConfig.zo_random_seed)
        for name, param in module.named_parameters():
            if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * mezoConfig.zo_eps

    def _zo_module_clone(self, module:nn.Module):
        cloned_module = deepcopy(module)
        return cloned_module

    def _zo_update(self, module:nn.Module):
        for name, param in module.named_parameters():
            # Resample z
            torch.manual_seed(mezoConfig.zo_random_seed)
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if param.requires_grad:
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + trainConfig.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

    def _zo_zero_grad(self):
        self.projected_grad = 0

    def _get_learning_rate(self):
        return trainConfig.max_lr


class GPT2ModelMezoOffloading(GPT2ModelMezo):

    def __init__(self, config: ModelConfig, cuda_device="cuda:0"):
        super().__init__(config)
        self.overlap = True
        self.empty_cache_every_layers = 1
        self.mezo_init(device=cuda_device)
        self.mezo_error_handler()
    
    # Offloading added
    def mezo_init(self, device="cuda:0"):
        self.transformer.wte = self.transformer.wte.to(device)
        self.transformer.wpe = self.transformer.wpe.to(device)
        self.transformer.ln_f = self.transformer.ln_f.to(device)
        self.lm_head = self.lm_head.to(device)
    
    # Offloading added
    def mezo_error_handler(self, alpha=0.5):
        block_size = 0
        model_size = 0
        for p in self.transformer.h.parameters():
            block_size += p.numel()
        for p in self.parameters():
            model_size += p.numel()
        if block_size / model_size < (1-alpha):
            raise ValueError(f"transformer blocks should be greater than {(1-alpha)*100}%")
    
    def forward(self, idx, targets=None):
        if self.zo_training:
            return self.zo_forward(idx, targets)
        # idx is of shape (B, T)
        B, T = idx.size()
        device = idx.device
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Offloading added: pre load one block
        self.cpu_offload_stream = torch.cuda.Stream()
        self.cpu_upload_stream = torch.cuda.Stream()
        block = self.transformer.h[0]
        block = self.uploading(block, device)
        
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for i in range(1, self.config.n_layer):
            # Offloading added: next block CPU uploading
            self.transformer.h[i] = self.uploading(self.transformer.h[i], device)
            
            # block forward
            x = block(x)

            # Offloading added: CPU offloading
            block = self.offloading(block)
            
            block = self.transformer.h[i]
            if i%self.empty_cache_every_layers==0:
                torch.cuda.empty_cache()
        x = block(x)

        # Offloading added: CPU offloading
        block = self.offloading(block)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, pad_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Offloading added: sync all CPU offloading
        self.cpu_offload_stream.synchronize()
        torch.cuda.empty_cache()

        return logits, loss
    
    ############## Uploading / Offloading ##############

    def uploading(self, module: nn.Module, device: torch.device):
        if self.overlap:
            self.cpu_upload_stream.synchronize()
            with torch.cuda.stream(self.cpu_upload_stream):
                module = module.to(device, non_blocking=True)
        else:
            module = module.to(device)
        return module

    def offloading(self, module: nn.Module):
        if self.overlap:
            self.cpu_offload_stream.synchronize()
            with torch.cuda.stream(self.cpu_offload_stream):
                module = module.to("cpu", non_blocking=True)
        else:
            module = module.to("cpu")
        return module

    ############## MeZO ##############

    def zo_forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        device = idx.device
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Offloading added: pre load one block
        self.cpu_offload_stream = torch.cuda.Stream()
        self.cpu_upload_stream = torch.cuda.Stream()
        block = self.transformer.h[0]
        block = self.uploading(block, device)
        
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
        pos_emb1, pos_emb2 = self.zo_dual_forward(self.transformer.wpe, (pos, pos))
        tok_emb1, tok_emb2 = self.zo_dual_forward(self.transformer.wte, (idx, idx))
        with torch.inference_mode():
            x1, x2 = tok_emb1 + pos_emb1, tok_emb2 + pos_emb2
        
        # forward the blocks of the transformer
        for i in range(1, self.config.n_layer):
            # Offloading added: next block CPU uploading
            self.transformer.h[i] = self.uploading(self.transformer.h[i], device)
            
            # block fual forward
            x1, x2 = self.zo_dual_forward(block, (x1, x2))

            # Offloading added: CPU offloading
            block = self.offloading(block)
            
            # update block
            block = self.transformer.h[i]
            if i%self.empty_cache_every_layers==0:
                torch.cuda.empty_cache()

        # block fual forward
        x1, x2 = self.zo_dual_forward(block, (x1, x2))

        # Offloading added: CPU offloading
        block = self.offloading(block)

        # forward the final layernorm and the classifier
        x1, x2 = self.zo_dual_forward(self.transformer.ln_f, (x1, x2))
        logits1, logits2 = self.zo_dual_forward(self.lm_head, (x1, x2))
        loss1 = loss2 = None
        if targets is not None:
            with torch.inference_mode():
                loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets.view(-1))
                loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), targets.view(-1))
                self.zo_final_step(loss1, loss2)
        
        # Offloading added: sync all CPU offloading
        self.cpu_offload_stream.synchronize()
        torch.cuda.empty_cache()

        return (logits1, logits2), (loss1, loss2)
    

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_peak_memory_usage(iter, device="cuda:0", use_tqdm=False):
    # Check the peak memory usage
    peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
    if use_tqdm:
        tqdm.write("Peak GPU Memory after iteration {}: {:.2f} MB".format(iter+1, peak_memory))
    else:
        print(f"Peak GPU Memory after iteration {iter+1}: {peak_memory:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device=device)

def check_time_cost(iter, fn, *args, use_tqdm=False, **kwargs):
    t1 = time.time()
    out = fn(*args, **kwargs)
    t2 = time.time()
    time_cost = t2-t1
    throughtput = trainConfig.total_token_batch_size / time_cost
    if use_tqdm:
        tqdm.write("Time cost after iteration {}: {:.2f} ms, {:.2f} tok/s".format(iter+1, time_cost*1e3, throughtput))
    else:
        print("Time cost after iteration {}: {:.2f} ms, {:.2f} tok/s".format(iter+1, time_cost*1e3, throughtput))

def model_size(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def eval_acc():
    seed_everything(42)
    model_ref = GPT2ModelMezo(modelConfig)
    print(f"normal model size: {model_size(model_ref)/1024**3:.2f} B")
    model = GPT2ModelMezoOffloading(modelConfig, cuda_device=modelConfig.device)
    print(f"Offloading model size: {model_size(model)/1024**3:.2f} B")
    for name_ref, p_ref in model_ref.named_parameters():
        for name, p in model.named_parameters():
            if name==name_ref:
                # print(name)
                p_ref.data.copy_(p.data)
    model_ref = model_ref.to(modelConfig.device)

    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}

    out_ref = model_ref(**input)
    out = model(**input)
    (output1_ref, output2_ref), (loss1_ref, loss2_ref) = out_ref[:2]
    (output1, output2), (loss1, loss2) = out[:2]

    print(loss1_ref.item(), loss1.item())
    print(loss2_ref.item(), loss2.item())
    print(torch.allclose(loss1_ref, loss1))
    print(torch.allclose(loss2_ref, loss2))
    print("diff loss1: ", torch.abs(loss1_ref - loss1).max())
    print("diff loss2: ", torch.abs(loss2_ref - loss2).max())
    print("diff output1: ", torch.abs(output1_ref - output1).max())
    print("diff output2: ", torch.abs(output2_ref - output2).max())

def mezo_performance():
    seed_everything(42)
    model_ref = GPT2ModelMezo(modelConfig)
    print(f"normal model size: {model_size(model_ref)/1024**3:.2f} B")
    model_ref = model_ref.to(modelConfig.device)
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(trainConfig.max_steps)):
        check_time_cost(i, model_ref, **input)
        check_peak_memory_usage(i, modelConfig.device, True)

def mezo_offloading_performance():
    seed_everything(42)
    model = GPT2ModelMezoOffloading(modelConfig, cuda_device=modelConfig.device)
    print(f"Offloading model size: {model_size(model)/1024**3:.2f} B")
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(trainConfig.max_steps)):
        check_time_cost(i, model, **input)
        check_peak_memory_usage(i, modelConfig.device, True)

def train_mezo():
    seed_everything(42)
    model_ref = GPT2ModelMezo(modelConfig)
    print(f"normal model size: {model_size(model_ref)/1024**3:.2f} B")
    model_ref = model_ref.to(modelConfig.device)
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    for i in tqdm(range(trainConfig.max_steps)):
        # train
        model_ref.zo_training = True
        model_ref.grad_accum = False
        model_ref(**input)

        # eval
        model_ref.zo_training = False
        loss = model_ref(**input)[-1]
        tqdm.write("Iteration {}, loss: {}".format(i, loss))

def train_mezo_offloading():
    seed_everything(42)
    model = GPT2ModelMezoOffloading(modelConfig, cuda_device=modelConfig.device)
    print(f"normal model size: {model_size(model)/1024**3:.2f} B")
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    for i in tqdm(range(trainConfig.max_steps)):
        # train
        model.zo_training = True
        model.grad_accum = False
        model(**input)

        # eval
        model.zo_training = False
        loss = model(**input)[-1]
        tqdm.write("Iteration {}, loss: {}".format(i, loss))


if __name__=="__main__":
    modelConfig = ModelConfig()
    trainConfig = TrainConfig()
    mezoConfig = MezoConfig()

    # eval_acc()
    # mezo_performance()
    # mezo_offloading_performance()

    # train_mezo()
    train_mezo_offloading()
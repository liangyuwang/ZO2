
import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig, MezoConfig
from .nanogpt_mezo import GPT2ModelMezo


class GPT2ModelMezoOffloading(GPT2ModelMezo):

    def __init__(self, config: ModelConfig, mezoConfig: MezoConfig, cuda_device="cuda:0"):
        super().__init__(config, mezoConfig)
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
    

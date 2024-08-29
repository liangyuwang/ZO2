
import torch
import torch.nn as nn
from torch.nn import functional as F

from mezo_offload import BaseMezoOffloadingModel

from .configs import ModelConfig, MezoConfig, OffloadingConfig
from .nanogpt import Block


class GPT2ModelMezoOffloading(nn.Module, BaseMezoOffloadingModel):

    def __init__(self, config: ModelConfig, mezoConfig: MezoConfig, offloadingConfig: OffloadingConfig):
        super().__init__()
        self.config = config
        self.mezoConfig = mezoConfig
        self.offloadingConfig = offloadingConfig
        self.mezo_config()
        self.offloading_config()
        
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
    
        self.offloading_init(offloadingConfig)
        self.offloading_error_handler()
    
    def forward(self, idx, targets=None):
        if self.zo_training:
            return self.zo_forward(idx, targets)
        # idx is of shape (B, T)
        B, T = idx.size()
        device = idx.device
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Offloading added: pre load one block
        block = self.transformer.h[0]
        block = self.uploading(block, sync=False)
        
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for i in range(1, self.config.n_layer):
            # Offloading added: next block CPU uploading
            if i in self.upload_next_layer_start_ids:
                next_upload_layer_id = self.offload_layer_ids[self.uploaded_layer_idx_counter+1]
                self.transformer.h[next_upload_layer_id] = self.uploading(self.transformer.h[next_upload_layer_id])
            
            # last transformer block forward
            if self.check_fully_uploaded(i-1):
                x = block(x)

            # Offloading added: CPU offloading
            if (i-1) in self.offload_layer_ids:
                block = self.offloading(block)
            
            # update block
            block = self.transformer.h[i]
            if (i-1) % self.empty_cache_every_blocks==0:
                torch.cuda.empty_cache()
        
        # Offloading added: sync final CPU uploading
        if self.overlap and self.config.n_layer-1 in self.offload_layer_ids:
            self.upload_stream.synchronize()
            self.mark_fully_uploaded(self.offload_layer_ids[self.uploaded_layer_idx_counter])
        
        # final block forward
        if self.check_fully_uploaded(self.config.n_layer-1):
            x = block(x)

        # Offloading added: final CPU offloading
        if (self.config.n_layer-1) in self.offload_layer_ids:
            block = self.offloading(block)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, pad_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Offloading added: sync final CPU offloading
        if self.overlap and self.config.n_layer-1 in self.offload_layer_ids:
            self.offload_stream.synchronize()
        torch.cuda.empty_cache()

        # reset offloading state
        self.reset_offloading()

        return logits, loss
    
    ############## Uploading / Offloading ##############

    def offloading_config(self):
        self.offload_to_device = self.offloadingConfig.offload_to_device
        self.offload_from_device = self.offloadingConfig.offload_from_device
        self.overlap = self.offloadingConfig.overlap
        self.offload_every_blocks = self.offloadingConfig.offload_every_blocks
        self.empty_cache_every_blocks = self.offloadingConfig.empty_cache_every_blocks
        self.offload_use_amp = self.offloadingConfig.offload_use_amp
        self.offload_amp_dtype = self.offloadingConfig.offload_amp_dtype
        self.medium_precision_blocks_on_device = self.offloadingConfig.medium_precision_blocks_on_device
        self.offloading_args(n_layer=self.config.n_layer)

    def offloading_init(self, offloadingConfig: OffloadingConfig):
        self.transformer.wte = self.transformer.wte.to(offloadingConfig.offload_from_device)
        self.transformer.wpe = self.transformer.wpe.to(offloadingConfig.offload_from_device)
        self.transformer.ln_f = self.transformer.ln_f.to(offloadingConfig.offload_from_device)
        self.lm_head = self.lm_head.to(offloadingConfig.offload_from_device)
        for i in range(len(self.transformer.h)):
            if i not in self.offload_layer_ids:
                self.transformer.h[i] = self.transformer.h[i].to(offloadingConfig.offload_from_device)
                if offloadingConfig.offload_use_amp and self.medium_precision_blocks_on_device:
                    self.transformer.h[i] = self.transformer.h[i].to(offloadingConfig.offload_amp_dtype)
            else:
                if offloadingConfig.offload_use_amp:
                    self.transformer.h[i] = self.transformer.h[i].to(offloadingConfig.offload_amp_dtype)

    def offloading_error_handler(self, alpha=0.5):
        block_size = sum(p.numel() for p in self.transformer.h.parameters())
        model_size = sum(p.numel() for p in self.parameters())
        if block_size / model_size < (1-alpha):
            raise ValueError(f"Transformer blocks' parameters should be greater than {(1-alpha)*100}%")
    
    ############## MeZO ##############

    def mezo_config(self):
        self.max_zo_random_seed = self.mezoConfig.max_zo_random_seed
        self.zo_eps = self.mezoConfig.zo_eps
        self.non_diff = self.mezoConfig.non_diff
        self.zo_lr = self.mezoConfig.zo_lr
        self.zo_weight_decay = self.mezoConfig.zo_weight_decay
        self.mezo_args()

    @torch.inference_mode
    def zo_forward(self, idx, targets=None):
        self.set_random_seed()
        # idx is of shape (B, T)
        B, T = idx.size()
        device = idx.device
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Offloading added: pre load one block
        block = self.transformer.h[0]
        block = self.uploading(block, sync=False)
        
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
        pos_emb1, pos_emb2 = self.zo_dual_forward(self.transformer.wpe, (pos, pos))
        tok_emb1, tok_emb2 = self.zo_dual_forward(self.transformer.wte, (idx, idx))
        x1, x2 = tok_emb1 + pos_emb1, tok_emb2 + pos_emb2
        
        # forward the blocks of the transformer
        for i in range(1, self.config.n_layer):
            # Offloading added: next block CPU uploading
            if i in self.upload_next_layer_start_ids:
                next_upload_layer_id = self.offload_layer_ids[self.uploaded_layer_idx_counter+1]
                self.transformer.h[next_upload_layer_id] = self.uploading(self.transformer.h[next_upload_layer_id])
            
            # last transformer block dual forward
            if self.check_fully_uploaded(i-1):
                x1, x2 = self.zo_dual_forward(block, (x1, x2), amp_cast=True)

            # Offloading added: CPU offloading
            if (i-1) in self.offload_layer_ids:
                block = self.offloading(block)
            
            # update block
            block = self.transformer.h[i]
            if i%self.empty_cache_every_blocks==0:
                torch.cuda.empty_cache()

        # Offloading added: sync final CPU uploading
        if self.overlap and self.config.n_layer-1 in self.offload_layer_ids:
            self.upload_stream.synchronize()
            self.mark_fully_uploaded(self.offload_layer_ids[self.uploaded_layer_idx_counter])
        
        # block final dual forward
        if self.check_fully_uploaded(self.config.n_layer-1):
            x1, x2 = self.zo_dual_forward(block, (x1, x2), amp_cast=True)

        # Offloading added: final CPU offloading
        if (self.config.n_layer-1) in self.offload_layer_ids:
            block = self.offloading(block)

        # forward the final layernorm and the classifier
        x1, x2 = self.zo_dual_forward(self.transformer.ln_f, (x1, x2))
        logits1, logits2 = self.zo_dual_forward(self.lm_head, (x1, x2))
        loss1 = loss2 = None
        if targets is not None:
            loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets.view(-1))
            loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), targets.view(-1))
            self.zo_final_step(loss1, loss2)
        
        # Offloading added: sync final CPU offloading
        if self.overlap and self.config.n_layer-1 in self.offload_layer_ids:
            self.offload_stream.synchronize()
        torch.cuda.empty_cache()

        # reset offloading state
        self.reset_offloading()

        return (logits1, logits2), (loss1, loss2)
    

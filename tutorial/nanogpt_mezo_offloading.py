
import torch
import torch.nn as nn
from torch.nn import functional as F

from .configs import ModelConfig, MezoConfig, OffloadingConfig
from .nanogpt_mezo import GPT2ModelMezo


class GPT2ModelMezoOffloading(GPT2ModelMezo):

    def __init__(self, config: ModelConfig, mezoConfig: MezoConfig, offloadingConfig: OffloadingConfig):
        self.offloading_args(offloadingConfig, config.n_layer)
        super().__init__(config, mezoConfig)
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
        self.offload_stream = torch.cuda.Stream()
        self.upload_stream = torch.cuda.Stream()
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
            if (i-1) % self.offloadingConfig.empty_cache_every_blocks==0:
                torch.cuda.empty_cache()
        
        # Offloading added: sync final CPU uploading
        if self.offloadingConfig.overlap and self.config.n_layer-1 in self.offload_layer_ids:
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
        if self.offloadingConfig.overlap and self.config.n_layer-1 in self.offload_layer_ids:
            self.offload_stream.synchronize()
        torch.cuda.empty_cache()

        # reset offloading state
        self.reset_offloading()

        return logits, loss
    
    ############## Uploading / Offloading ##############

    def offloading_args(self, offloadingConfig: OffloadingConfig, n_layer):
        self.offloadingConfig = offloadingConfig
        if self.offloadingConfig.offload_every_blocks >= n_layer:
            raise ValueError("The 'offloadingConfig.offload_every_blocks' must smaller than number of layers")
        self.offloadingConfig.empty_cache_every_blocks = max(self.offloadingConfig.empty_cache_every_blocks, self.offloadingConfig.offload_every_blocks)
        self.offload_layer_ids = list(range(0, n_layer, self.offloadingConfig.offload_every_blocks)) # which layers should be offloaded
        if self.offload_layer_ids[0] != 0:
            raise ValueError(f"Transformer block 0 must be offloaded to {self.offloadingConfig.offload_to_device}.")
        self.upload_next_layer_start_ids = [i+1 for i in self.offload_layer_ids]    # which time you want next layer to be pre-uploaded
        self.upload_next_layer_start_ids.pop(-1)
        self.uploaded_layer_idx_counter = -1    # index of 'self.offload_layer_ids', showing which layers are already uploaded
        self.if_layers_fully_uploaded = {
            "init": [False if i in self.offload_layer_ids else True for i in range(n_layer)],
            "runtime": [False if i in self.offload_layer_ids else True for i in range(n_layer)]
        }
        print(f"Transformer blocks {self.offload_layer_ids} will be offloaded to {self.offloadingConfig.offload_to_device}")

    def offloading_init(self, offloadingConfig: OffloadingConfig):
        self.transformer.wte = self.transformer.wte.to(offloadingConfig.offload_from_device)
        self.transformer.wpe = self.transformer.wpe.to(offloadingConfig.offload_from_device)
        self.transformer.ln_f = self.transformer.ln_f.to(offloadingConfig.offload_from_device)
        self.lm_head = self.lm_head.to(offloadingConfig.offload_from_device)
        for i in range(len(self.transformer.h)):
            if i not in self.offload_layer_ids:
                self.transformer.h[i] = self.transformer.h[i].to(offloadingConfig.offload_from_device)

    def offloading_error_handler(self, alpha=0.5):
        block_size = sum(p.numel() for p in self.transformer.h.parameters())
        model_size = sum(p.numel() for p in self.parameters())
        if block_size / model_size < (1-alpha):
            raise ValueError(f"Transformer blocks' parameters should be greater than {(1-alpha)*100}%")
    
    def mark_fully_uploaded(self, layer_id):
        self.if_layers_fully_uploaded["runtime"][layer_id] = True

    def check_fully_uploaded(self, layer_id):
        if self.offloadingConfig.overlap:
            if self.if_layers_fully_uploaded["runtime"][layer_id]:
                return True
            else:
                raise ValueError(f"Transformer block {layer_id} is not fully uploaded from {self.offloadingConfig.offload_to_device}")
        return True

    def uploading(self, module: nn.Module, sync: bool=True):
        if self.offloadingConfig.overlap:
            if sync:
                self.upload_stream.synchronize()
            self.uploaded_layer_idx_counter += 1
            self.mark_fully_uploaded(self.offload_layer_ids[self.uploaded_layer_idx_counter])
            with torch.cuda.stream(self.upload_stream):
                module = module.to(self.offloadingConfig.offload_from_device, non_blocking=True)
        else:
            module = module.to(self.offloadingConfig.offload_from_device)
            self.uploaded_layer_idx_counter += 1
        return module

    def offloading(self, module: nn.Module):
        if self.offloadingConfig.overlap:
            self.offload_stream.synchronize()
            with torch.cuda.stream(self.offload_stream):
                module = module.to(self.offloadingConfig.offload_to_device, non_blocking=True)
        else:
            module = module.to(self.offloadingConfig.offload_to_device)
        return module
    
    def reset_offloading(self):
        self.if_layers_fully_uploaded["runtime"] = self.if_layers_fully_uploaded["init"]
        self.uploaded_layer_idx_counter = -1

    ############## MeZO ##############

    @torch.inference_mode
    def zo_forward(self, idx, targets=None):
        self.set_random_seed()
        # idx is of shape (B, T)
        B, T = idx.size()
        device = idx.device
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Offloading added: pre load one block
        self.offload_stream = torch.cuda.Stream()
        self.upload_stream = torch.cuda.Stream()
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
                x1, x2 = self.zo_dual_forward(block, (x1, x2))

            # Offloading added: CPU offloading
            if (i-1) in self.offload_layer_ids:
                block = self.offloading(block)
            
            # update block
            block = self.transformer.h[i]
            if i%self.offloadingConfig.empty_cache_every_blocks==0:
                torch.cuda.empty_cache()

        # Offloading added: sync final CPU uploading
        if self.offloadingConfig.overlap and self.config.n_layer-1 in self.offload_layer_ids:
            self.upload_stream.synchronize()
            self.mark_fully_uploaded(self.offload_layer_ids[self.uploaded_layer_idx_counter])
        
        # block final dual forward
        if self.check_fully_uploaded(self.config.n_layer-1):
            x1, x2 = self.zo_dual_forward(block, (x1, x2))

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
        if self.offloadingConfig.overlap and self.config.n_layer-1 in self.offload_layer_ids:
            self.offload_stream.synchronize()
        torch.cuda.empty_cache()

        # reset offloading state
        self.reset_offloading()

        return (logits1, logits2), (loss1, loss2)
    

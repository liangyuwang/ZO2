
import os
import gc
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F

from .mezo import BaseMezoModel


class BaseMezoOffloadingModel(BaseMezoModel):

    def __init__(self):
        self.set_mezo_config()
        self.set_offloading_config()
        self.offloading_reinit()
    
    ############## Uploading / Offloading ##############

    def set_offloading_config(self):
        self.offload_to_device: torch.device = "cpu"
        self.offload_from_device: torch.device = "cuda:0"
        self.overlap: bool = True    # if you want to make communication-computation overlap, 'True' will be faster.
        self.offload_every_blocks: int = 1   # how many layers per interval do you want to offload a layer
        self.empty_cache_every_blocks: int = 1   # frequency of empty cache
        self.offload_use_amp: bool = True
        self.offload_amp_dtype: torch.dtype = torch.bfloat16
        self.medium_precision_blocks_on_device: bool = True
        self.set_offloading_args(n_layer=...)

    def set_offloading_args(self, n_layer):
        if self.offload_every_blocks >= n_layer:
            raise ValueError("The 'offload_every_blocks' must smaller than number of layers")
        self.empty_cache_every_blocks = max(self.empty_cache_every_blocks, self.offload_every_blocks)
        self.offload_layer_ids = list(range(0, n_layer, self.offload_every_blocks)) # which layers should be offloaded
        if self.offload_layer_ids[0] != 0:
            raise ValueError(f"Transformer block 0 must be offloaded to {self.offload_to_device}.")
        self.upload_next_layer_start_ids = [i+1 for i in self.offload_layer_ids]    # which time you want next layer to be pre-uploaded
        self.upload_next_layer_start_ids.pop(-1)
        self.uploaded_layer_idx_counter = 0    # index of 'self.offload_layer_ids', showing which layers are already uploaded
        self.if_layers_fully_uploaded = {
            "init": [False if i in self.offload_layer_ids else True for i in range(n_layer)],
            "runtime": [False if i in self.offload_layer_ids else True for i in range(n_layer)]
        }
        print(f"Transformer blocks {self.offload_layer_ids} will be offloaded to {self.offload_to_device}")
        self.offload_stream = torch.cuda.Stream()
        self.upload_stream = torch.cuda.Stream()

    def offloading_reinit(self):
        # init modules in the Model
        ...

    def mark_fully_uploaded(self, layer_id):
        self.if_layers_fully_uploaded["runtime"][layer_id] = True

    def check_fully_uploaded(self, layer_id):
        if self.overlap:
            if self.if_layers_fully_uploaded["runtime"][layer_id]:
                return True
            else:
                raise ValueError(f"Transformer block {layer_id} is not fully uploaded from {self.offload_to_device}")
        return True

    def uploading(self, module: nn.Module, sync: bool=True):
        if self.overlap:
            if sync:
                self.upload_stream.synchronize()
                self.uploaded_layer_idx_counter += 1
            self.mark_fully_uploaded(self.offload_layer_ids[self.uploaded_layer_idx_counter])
            with torch.cuda.stream(self.upload_stream):
                module = module.to(self.offload_from_device, non_blocking=True)
        else:
            if sync:
                self.uploaded_layer_idx_counter += 1
            module = module.to(self.offload_from_device)
        return module

    def offloading(self, module: nn.Module):
        if self.overlap:
            self.offload_stream.synchronize()
            with torch.cuda.stream(self.offload_stream):
                module = module.to(self.offload_to_device, non_blocking=True)
        else:
            module = module.to(self.offload_to_device)
        return module
    
    def reset_offloading(self):
        self.if_layers_fully_uploaded["runtime"] = self.if_layers_fully_uploaded["init"]
        self.uploaded_layer_idx_counter = 0

    @torch.inference_mode
    def zo_dual_forward(self, module:nn.Module, dual_inputs, update=True, amp_cast=False):
        input1, input2 = dual_inputs
        if (self.projected_grad != 0 and not self.grad_accum) and update:
            self._zo_update(module)
        cloned_module = self._module_clone(module)
        self._zo_perturb_parameters(cloned_module, scaling_factor=1)
        out1 = cloned_module(input1)
        self._zo_perturb_parameters(cloned_module, scaling_factor=-2)
        out2 = cloned_module(input2)
        del cloned_module
        if self.offload_use_amp and amp_cast:
            module = module.to(self.offload_amp_dtype)
        return out1, out2
    
    @torch.inference_mode
    def _module_clone(self, module:nn.Module):
        cloned_module = deepcopy(module)
        return cloned_module

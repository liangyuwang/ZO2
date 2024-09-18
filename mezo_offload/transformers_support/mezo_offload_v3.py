
import os
import gc
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F

from .. import mezo_offload_v3


class BaseMezoOffloadingModel(mezo_offload_v3.BaseMezoOffloadingModel):

    @torch.inference_mode
    def zo_dual_forward(self, module:nn.Module, dual_inputs, update=True, amp_cast=False):
        input1, input2 = dual_inputs
        if self.offload_use_amp and amp_cast:
            module = self.compress_decode(module)
        if (self.projected_grad != 0 and not self.grad_accum) and update:
            self._zo_update(module)
        cloned_module = self._module_clone(module)
        self._zo_perturb_parameters(cloned_module, scaling_factor=1)
        if self.offload_use_amp and amp_cast:
            with torch.autocast("cuda", self.offload_amp_dtype):
                out1 = cloned_module(**input1)
        else:
            out1 = cloned_module(**input1)
        self._zo_perturb_parameters(cloned_module, scaling_factor=-2)
        if self.offload_use_amp and amp_cast:
            with torch.autocast("cuda", self.offload_amp_dtype):
                out2 = cloned_module(**input2)
        else:
            out2 = cloned_module(**input2)
        if self.offload_use_amp and amp_cast:
            module = self.compress_encode(module)
        return out1, out2
    
    @torch.inference_mode
    def zo_final_step(self, loss1, loss2):
        super().zo_final_step(loss1, loss2)
        self.set_random_seed()
    

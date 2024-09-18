
import os
import gc
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F

from .. import mezo_offload_v1


class BaseMezoOffloadingModel(mezo_offload_v1.BaseMezoOffloadingModel):

    @torch.inference_mode
    def zo_dual_forward(self, module:nn.Module, dual_inputs, update=True, amp_cast=False):
        input1, input2 = dual_inputs
        if (self.projected_grad != 0 and not self.grad_accum) and update:
            self._zo_update(module)
        cloned_module = self._module_clone(module)
        self._zo_perturb_parameters(cloned_module, scaling_factor=1)
        out1 = cloned_module(**input1)
        self._zo_perturb_parameters(cloned_module, scaling_factor=-2)
        out2 = cloned_module(**input2)
        del cloned_module
        if self.offload_use_amp and amp_cast:
            module = module.to(self.offload_amp_dtype)
        return out1, out2
    
    @torch.inference_mode
    def zo_final_step(self, loss1, loss2):
        super().zo_final_step(loss1, loss2)
        self.set_random_seed()
    

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .. import mezo


class BaseMezoModel(mezo.BaseMezoModel):

    @torch.inference_mode
    def zo_dual_forward(self, module:nn.Module, dual_inputs, update=True):
        input1, input2 = dual_inputs
        if (self.projected_grad != 0 and not self.grad_accum) and update:
            self._zo_update(module)
        self._zo_perturb_parameters(module, scaling_factor=1)
        out1 = module(**input1)
        self._zo_perturb_parameters(module, scaling_factor=-2)
        out2 = module(**input2)
        # Reset model back to its parameters at start of step (1-2+1=0)
        self._zo_perturb_parameters(module, scaling_factor=1)
        return out1, out2
    
    @torch.inference_mode
    def zo_final_step(self, loss1, loss2):
        super().zo_final_step(loss1, loss2)
        self.set_random_seed()
    

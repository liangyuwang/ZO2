
import os
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseMezoModel:

    def __init__(self):
        self.mezo_config()
    
    ############## MeZO ##############
    # inspired by https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py

    def mezo_config(self):
        self.max_zo_random_seed = 1000000000
        self.zo_eps = 1e-3
        self.non_diff = False
        self.zo_lr = 1e-7
        self.zo_weight_decay = 1e-1
        self.mezo_args()
    
    def mezo_args(self):
        self.zo_training = True
        self.projected_grad = 0
        self.grad_accum = False

    @torch.inference_mode
    def zo_dual_forward(self, module:nn.Module, dual_inputs, update=True):
        input1, input2 = dual_inputs
        if (self.projected_grad != 0 and not self.grad_accum) and update:
            self._zo_update(module)
            self._zo_zero_grad()
        self._zo_perturb_parameters(module, scaling_factor=1)
        out1 = module(input1)
        self._zo_perturb_parameters(module, scaling_factor=-2)
        out2 = module(input2)
        # Reset model back to its parameters at start of step (1-2+1=0)
        self._zo_perturb_parameters(module, scaling_factor=1)
        return out1, out2
    
    @torch.inference_mode
    def zo_final_step(self, loss1, loss2):
        self.projected_grad += (loss1 - loss2) / (2 * self.zo_eps)
    
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
                param.data = param.data + scaling_factor * z * self.zo_eps

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
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + self.zo_weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

    @torch.inference_mode
    def _zo_zero_grad(self):
        self.projected_grad = 0

    @torch.inference_mode
    def _get_learning_rate(self):
        return self.zo_lr

    @torch.inference_mode
    def set_random_seed(self):
        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(self.max_zo_random_seed)


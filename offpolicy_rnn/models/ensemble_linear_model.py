import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        bias: bool=True,
        desire_ndim: int=None
    ) -> None:
        super().__init__()
        self.use_bias = bias
        self.desire_ndim = desire_ndim
        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        if self.use_bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))
        self.device = torch.device('cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.use_bias:
            bias = self.bias
        else:
            bias = None
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        elif len(x.shape) == 3:
            # TODO: judge the shape carefully according to your application
            if (self.desire_ndim is None or self.desire_ndim == 3) and x.shape[0] == weight.data.shape[0]:
                x = torch.einsum('bij,bjk->bik', x, weight)
            else:
                x = torch.einsum('cij,bjk->bcik', x, weight)
        elif len(x.shape) == 4:
            if (self.desire_ndim is None or self.desire_ndim == 4) and x.shape[0] == weight.data.shape[0]:
                x = torch.einsum('cbij,cjk->cbik', x, weight)
            else:
                x = torch.einsum('cdij,bjk->bcdik', x, weight)
        elif len(x.shape) == 5:
            x = torch.einsum('bcdij,bjk->bcdik', x, weight)
        if self.use_bias:
            assert x.shape[0] == bias.shape[0] and x.shape[-1] == bias.shape[-1]
            if len(x.shape) == 4:
                bias = bias.unsqueeze(1)
            elif len(x.shape) == 5:
                bias = bias.unsqueeze(1)
                bias = bias.unsqueeze(1)

            x = x + bias

        return x

    def to(self, device):
        if not device == self.device:
            self.device = device
            super().to(device)
            self.weight = self.weight.to(self.device)
            if self.use_bias:
                self.bias = self.bias.to(self.device)


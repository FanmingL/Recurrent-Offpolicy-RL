import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class MultiEnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        multi_num: int,
        bias: bool=True,
        desire_ndim: int=None
    ) -> None:
        super().__init__()
        self.use_bias = bias
        self.desire_ndim = desire_ndim
        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(multi_num, num_ensemble, input_dim, output_dim)))
        if self.use_bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(multi_num, num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))
        self.device = torch.device('cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.desire_ndim is not None, 'for MLP network (data shape is (B, C)), desire_ndim should be 3, for RNN network, (data shape is (B, L, C)), desire_ndim should be 4, while got None'

        weight = self.weight
        if self.use_bias:
            bias = self.bias
        else:
            bias = None

        if len(x.shape) == 2:
            assert self.desire_ndim == 3
            x = torch.einsum('ij,cbjk->cbik', x, weight)
        elif len(x.shape) == 3:
            if self.desire_ndim == 3:
                # target is
                x = torch.einsum('bij,cbjk->cbik', x, weight)
            elif self.desire_ndim == 4:
                x = torch.einsum('hij,cbjk->cbhik', x, weight)
            else:
                raise NotImplementedError
        elif len(x.shape) == 4:
            if self.desire_ndim == 4:
                x = torch.einsum('bhij,cbjk->cbhik', x, weight)
            else:
                raise NotImplementedError
        elif len(x.shape) == 5:
            assert self.desire_ndim == 4
            x = torch.einsum('cbhij,cbjk->cbhik', x, weight)
        if bias is not None:
            if self.desire_ndim == 3:
                x = x + bias
            elif self.desire_ndim == 4:
                x = x + bias.unsqueeze(2)
            else:
                raise NotImplementedError
        return x

    def to(self, device):
        if not device == self.device:
            self.device = device
            super().to(device)
            self.weight = self.weight.to(self.device)
            if self.use_bias:
                self.bias = self.bias.to(self.device)


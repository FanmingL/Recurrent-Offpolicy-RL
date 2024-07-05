import time

import numpy as np
import torch
import torch.nn as nn
try:
    from .scan_triton.real_rnn_tie_input_gate import real_scan_tie_input_gate, real_scan_tie_input_gate_fused
    from .scan_triton.real_rnn_fast_pscan import pscan_fast
except Exception as _:
    pass
from .scan_triton.real_rnn_tie_input_gate_cpu import scan_cpu, scan_cpu_fuse
from ..ensemble_linear_model import EnsembleLinear
class GILRLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            factor=1,
            dropout=0.0,
            use_ff=True,
            batch_first=True,
        ):
        super().__init__()
        assert batch_first
        self.d_model = output_dim
        self.in_proj = EnsembleLinear(input_dim, self.d_model * factor, 2, desire_ndim=4)
        self.out_proj = torch.nn.Linear(self.d_model * factor, self.d_model * factor)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(factor * self.d_model)
        self.swish =  nn.SiLU()
        self.use_ff = use_ff
        if self.use_ff:
            self.ff = PositionWiseFeedForward(self.d_model, dropout)
        self.device = torch.device('cpu')

    def rnn_parameters(self):
        return list(self.parameters(True))

    def to(self, device):
        if not self.device == device:
            super().to(device)
            self.device = device

    def forward(self, x, hidden=None, rnn_start=None):
        u = self.in_proj(x)
        v = u[0]
        f = u[1]
        if hidden is None:
            hidden = torch.zeros((v.shape[0], 1, self.d_model * 2), device=v.device)
        else:
            hidden = hidden.transpose(0, 1)
        hidden_pre = hidden
        f = torch.sigmoid(f)
        v = torch.tanh(v)
        if rnn_start is not None:
            f = f * (1 - rnn_start)
        if torch.all(hidden_pre == 0) and not self.device == torch.device('cpu'):
            v = real_scan_tie_input_gate(v.contiguous(), f.contiguous())
            hidden_pre = v[:, -1:, :]
        else:
            v, hidden_pre = scan_cpu(v, f, hidden_pre)
        out = self.out_proj(v)
        hidden = hidden_pre
        hidden = hidden.transpose(0, 1)
        if self.use_ff:
            out = self.ff(out)
        return out, hidden


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)

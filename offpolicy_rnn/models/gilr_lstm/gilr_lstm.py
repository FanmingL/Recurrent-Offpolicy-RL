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
class GILRLSTMLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            factor=1,
            dropout=0.2,
            batch_first=True,
        ):
        super().__init__()
        assert batch_first
        self.d_model = output_dim
        self.in_proj = EnsembleLinear(input_dim, self.d_model * factor, 2, desire_ndim=4)
        self.middle_proj = EnsembleLinear(self.d_model * factor, self.d_model * factor, 4, desire_ndim=4)
        self.out_proj = torch.nn.Linear(self.d_model * factor, self.d_model * factor)
        self.layer_norm = nn.LayerNorm(factor * self.d_model)
        self.swish =  nn.SiLU()
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
        hidden_pre, hidden_middle = torch.chunk(hidden, 2, -1)
        f = torch.sigmoid(f)
        v = torch.tanh(v)
        if rnn_start is not None:
            f = f * (1 - rnn_start)
        if torch.all(hidden_pre == 0) and not self.device == torch.device('cpu'):
            v = real_scan_tie_input_gate(v.contiguous(), f.contiguous())
            hidden_pre = v[:, -1:, :]
        else:
            v, hidden_pre = scan_cpu(v, f, hidden_pre)
        u = self.middle_proj.forward(v)
        f = torch.sigmoid(u[0])
        i = torch.sigmoid(u[1])
        o = torch.sigmoid(u[2])
        z = torch.tanh(u[3])
        if rnn_start is not None:
            f = f * (1 - rnn_start)
        if torch.all(hidden_middle == 0) and not self.device == torch.device('cpu'):
            v = i * z
            out = real_scan_tie_input_gate(v.contiguous(), f.contiguous())
            hidden_middle = out[:, -1:, :]
        else:
            out, hidden_middle = scan_cpu(i * z, f, hidden_middle)
        out = out * o
        out = self.out_proj(out)
        hidden = torch.cat((hidden_pre, hidden_middle), dim=-1)
        hidden = hidden.transpose(0, 1)
        return out, hidden

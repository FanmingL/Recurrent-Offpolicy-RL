import time

import numpy as np
import torch
import torch.nn as nn
try:
    from .scan_triton.real_rnn_tie_input_gate import real_scan_tie_input_gate, real_scan_tie_input_gate_fused
    from .scan_triton.real_rnn_fast_pscan import pscan_fast
except Exception as _:
    pass
    # import traceback
    # traceback.print_exc()
from .scan_triton.real_rnn_tie_input_gate_cpu import scan_cpu, scan_cpu_fuse
from ..ensemble_linear_model import EnsembleLinear
from ..multi_ensemble_linear_model import MultiEnsembleLinear
class EnsembleGILRLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_ensemble,
            factor=1,
            dropout=0.0,
            use_ff=True,
            batch_first=True,
        ):
        super().__init__()
        assert batch_first
        self.d_model = output_dim
        self.num_ensemble = num_ensemble
        self.in_proj = MultiEnsembleLinear(input_dim, self.d_model*factor, self.num_ensemble, 2, desire_ndim=4)
        # self.middle_proj = MultiEnsembleLinear(self.d_model*factor, self.d_model*factor, self.num_ensemble, 4, desire_ndim=4)
        self.out_proj = EnsembleLinear(self.d_model * factor, self.d_model * factor, self.num_ensemble)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([self.num_ensemble, factor * self.d_model])
        self.swish =  nn.SiLU()
        self.use_ff = use_ff
        if self.use_ff:
            self.ff = EnsemblePositionWiseFeedForward(self.d_model, num_ensemble, dropout, desire_ndim=4)
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

        f = torch.sigmoid(f)
        v = torch.tanh(v)
        if rnn_start is not None:
            if len(rnn_start.shape) < len(f.shape):
                f = f * (1 - rnn_start.unsqueeze(0))
            else:
                f = f * (1 - rnn_start)
        v = v.reshape((-1, v.shape[-2], v.shape[-1]))
        f = f.reshape((-1, f.shape[-2], f.shape[-1]))
        hidden_pre = hidden
        if (hidden_pre is None or torch.all(hidden_pre == 0)) and not self.device == torch.device('cpu'):
            # v = v.contiguous()
            # f = f.contiguous()
            # v = pscan_fast(v, f)
            v = real_scan_tie_input_gate(v.contiguous(), f.contiguous())
            hidden_pre = v[:, -1:, :]
        else:
            if hidden_pre is None:
                hidden_pre = torch.zeros((v.shape[0], 1, v.shape[-1]), device=v.device)
            else:
                hidden_pre = hidden_pre.transpose(0, 1)
            v, hidden_pre = scan_cpu(v, f, hidden_pre)
        out = v.reshape((self.num_ensemble, -1, v.shape[-2], v.shape[-1]))
        out = self.out_proj(out)
        hidden = hidden_pre
        hidden = hidden.transpose(0, 1)
        if self.use_ff:
            out = self.ff(out)
        return out, hidden


class EnsemblePositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, n_ensemble, dropout=0.1, desire_ndim=4):
        super().__init__()
        self.w_1 = EnsembleLinear(d_model, d_model, n_ensemble, desire_ndim=desire_ndim)
        self.w_2 = EnsembleLinear(d_model, d_model, n_ensemble, desire_ndim=desire_ndim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([n_ensemble, d_model])

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        x_ = self.dropout(self.w_2(x_)) + x
        return self.layer_norm(x_.transpose(0, -2)).transpose(0, -2)

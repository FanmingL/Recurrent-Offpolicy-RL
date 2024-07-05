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
class EnsembleGILRLSTMLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_ensemble,
            factor=1,
            dropout=0.2,
            batch_first=True,
        ):
        super().__init__()
        assert batch_first
        self.d_model = output_dim
        self.num_ensemble = num_ensemble
        self.in_proj = MultiEnsembleLinear(input_dim, self.d_model*factor, self.num_ensemble, 2, desire_ndim=4)
        self.middle_proj = MultiEnsembleLinear(self.d_model*factor, self.d_model*factor, self.num_ensemble, 4, desire_ndim=4)
        self.out_proj = EnsembleLinear(self.d_model * factor, self.d_model * factor, self.num_ensemble)
        self.layer_norm = nn.LayerNorm([self.num_ensemble, factor * self.d_model])
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

        f = torch.sigmoid(f)
        v = torch.tanh(v)
        if rnn_start is not None:
            if len(rnn_start.shape) < len(f.shape):
                f = f * (1 - rnn_start.unsqueeze(0))
            else:
                f = f * (1 - rnn_start)
        v = v.reshape((-1, v.shape[-2], v.shape[-1]))
        f = f.reshape((-1, f.shape[-2], f.shape[-1]))
        if hidden is not None:
            hidden_pre, hidden_middle = torch.chunk(hidden, 2, dim=-1)
        else:
            hidden_pre = None
            hidden_middle = None
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
        v = v.reshape((self.num_ensemble, -1, v.shape[-2], v.shape[-1]))

        u = self.middle_proj.forward(v)
        # u = u.reshape((4, -1, u.shape[-2], u.shape[-1]))
        f = torch.sigmoid(u[0])
        i = torch.sigmoid(u[1])
        o = torch.sigmoid(u[2])
        z = torch.tanh(u[3])
        if rnn_start is not None:
            if len(rnn_start.shape) < len(f.shape):
                f = f * (1 - rnn_start.unsqueeze(0))
            else:
                f = f * (1 - rnn_start)
        f = f.reshape((-1, f.shape[-2], f.shape[-1]))
        i = i.reshape((-1, i.shape[-2], i.shape[-1]))
        o = o.reshape((-1, o.shape[-2], o.shape[-1]))
        z = z.reshape((-1, z.shape[-2], z.shape[-1]))
        if (hidden_middle is None or torch.all(hidden_middle == 0)) and not self.device == torch.device('cpu'):
            v = i * z
            out = real_scan_tie_input_gate(v.contiguous(), f.contiguous())
            hidden_middle = out[:, -1:, :]
        else:
            if hidden_middle is None:
                hidden_middle = torch.zeros((v.shape[0], 1, v.shape[-1]), device=v.device)
            else:
                hidden_middle = hidden_middle.transpose(0, 1)
            out, hidden_middle = scan_cpu(i * z, f, hidden_middle)
        out = out * o
        out = out.reshape((self.num_ensemble, -1, out.shape[-2], out.shape[-1]))
        out = self.out_proj(out)
        hidden = torch.cat((hidden_pre, hidden_middle), dim=-1)
        hidden = hidden.transpose(0, 1)
        return out, hidden

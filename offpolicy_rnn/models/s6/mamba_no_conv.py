from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from .selective_scan.cpu_scan import selective_scan_cpu
try:
    from .selective_scan.triton_scan import triton_selective_scan_sequential
except Exception as _:
    triton_selective_scan_sequential = None

class MambaResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        assert input_dim == output_dim

        d_model = output_dim
        self.mixer = MambaBlock(input_dim)
        self.norm = RMSNorm(d_model)

    def forward(self, x, hidden=None, rnn_start=None):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
        """
        output, hidden = self.mixer(self.norm(x), hidden, rnn_start)
        output = output + x
        return output, hidden

class MambaBlock(nn.Module):
    def __init__(self, d_model, bias=False, dt_rank='auto', expand=2, d_state=16):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        d_inner = int(expand * d_model)
        if dt_rank == 'auto':
            dt_rank = int(math.ceil(d_model / 16))
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        # conv_bias = True
        # d_conv = 4
        # self.conv1d = nn.Conv1d(
        #     in_channels=d_inner,
        #     out_channels=d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=d_inner,
        #     padding=d_conv - 1,
        # )
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self._init_dt_proj_weight()

        self.d_state = d_state
        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner).contiguous()
        self.desired_hidden_dim = d_inner * d_state
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def _init_dt_proj_weight(self, dt_scale=1.0, dt_max=0.1, dt_min=0.001, dt_init_floor=1e-4, dt_init='random'):
        with torch.no_grad():
            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_proj.bias._no_reinit = True

    def forward(self, x, hidden=None, rnn_start=None):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # x = rearrange(x, 'b l d_in -> b d_in l')
        # x = self.conv1d(x)[:, :, :l]
        # x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)
        y, hidden = self.ssm(x, hidden, rnn_start)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output, hidden

    def ssm(self, x, hidden, start):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            hidden: shape (b, 1, d_in * n)
            start: shape (b, l, 1)

        Returns:
            output: shape (b, l, d_in)
            hidden: shape (b, 1, d_in * n)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape
        batch_size, T, _ = x.shape

        if hidden is None:
            hidden = torch.zeros((batch_size, d_in, n), device=x.device, dtype=x.dtype)
        else:
            hidden = hidden.reshape((batch_size, d_in, n))
        if start is None:
            start = torch.zeros((batch_size, T, 1), device=x.device, dtype=x.dtype)

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        if triton_selective_scan_sequential is not None and not x.device == torch.device('cpu'):
            y, hidden = triton_selective_scan_sequential(x, delta, A, B, C, D, start, initial_state=hidden)
        else:
            y, hidden = selective_scan_cpu(x, delta, A, B, C, D, start, initial_state=hidden)
        hidden = hidden.view((batch_size, 1, -1))
        return y, hidden


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

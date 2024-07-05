import numpy as np
import torch
import torch.nn as nn
try:
    # from .scan_triton.complex_rnn_2 import complex_scan
    # from .scan_triton.complex_rnn_jax import complex_scan
    from .scan_triton.complex_rnn import complex_scan
except:
    pass
    # import traceback
    # traceback.print_exc()

from .scan_triton.complex_rnn_cpu import complex_scan_cpu
from ..ensemble_linear_model import EnsembleLinear


class LRULayer(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            dropout=0.0,
            batch_first=True,
            use_ff = True,
            squash_inproj = False,
        ):
        super().__init__()
        assert batch_first, f'LRU only support batch_first==True'
        self.d_model = output_dim
        self.in_proj = EnsembleLinear(input_dim, self.d_model, num_ensemble=3, desire_ndim=4, bias=True)
        self.middle_proj = EnsembleLinear(self.d_model, self.d_model, num_ensemble=2, desire_ndim=4, bias=True)
        self.dropout = torch.nn.Dropout(dropout)
        nu_log, theta_log, gamma_log = self.initializer()
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)), requires_grad=True)
        self.use_ff = use_ff
        self.squash_inproj = squash_inproj
        if self.use_ff:
            self.ff = PositionWiseFeedForward(self.d_model, dropout)
        self.device = torch.device('cpu')

    def rnn_parameters(self):
        return self.parameters(recurse=True)

    def to(self, device):
        if not self.device == device:
            super().to(device)
            self.device = device

    def initializer(self):
        #https://arxiv.org/pdf/2303.06349.pdf Sect.3.2.2
        r_min, r_max = 0.9, 0.999
        u1 = torch.rand(self.d_model)
        u2 = torch.rand(self.d_model)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        # r_min, r_max = 0.8, 0.99
        # u1 = np.random.random(self.d_model)
        # u2 = np.random.random(self.d_model)
        # nu_log = np.log(
        #     -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        # )
        # theta_log = np.log(u2 * np.pi * 2)
        # gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        
        return nu_log, theta_log, gamma_log

    def forward(self, x, hidden: torch.Tensor=None, rnn_start=None, grad_detach=None):
        # x: B, L, C
        # rnn_start: B, L, 1
        # hidden: 1, B, C * 2
        # u: 3, B, L, C
        u = self.in_proj(x)
        if self.squash_inproj:
            u = torch.tanh(u)
        # input_real: B, L, C
        input_real = u[0]
        # input_imag: B, L, C
        input_imag = u[1]
        # o: B, L, C
        o = u[2]
        # nu: C
        # nu = torch.exp(-torch.exp(self.nu_log))
        # # theta: C
        # theta = torch.exp(self.theta_log)
        # # gamma: C
        # gamma = torch.exp(self.gamma_log)
        #
        # # f_real: C
        # f_real = nu * torch.cos(theta)
        # # f_imag: C
        # f_imag = nu * torch.sin(theta)
        params = torch.exp(self.params_log)
        nu = params[0]
        theta = params[1]
        gamma = params[2]
        lamb = torch.exp(torch.complex(-nu, theta))
        f_real = lamb.real
        f_imag = lamb.imag
        # print(f'nu: {nu.shape}, gamma: {gamma.shape}, lamb: {lamb.shape}, f_real:{f_real.shape}, f_image: {f_imag.shape}. input_real: {input_real.shape}')
        # input_real: B, L, C
        input_real = gamma[None, None, :] * input_real
        # input_imag: B, L, C
        input_imag = gamma[None, None, :] * input_imag
        # f_real: B, L, C
        f_real = f_real[None, None, :].expand_as(input_real)
        # f_real: B, L, C
        f_imag = f_imag[None, None, :].expand_as(input_imag)

        # rnn_start: B, L, 1
        if rnn_start is not None:
            f_real = f_real * (1 - rnn_start)
            f_imag = f_imag * (1 - rnn_start)
        if grad_detach is None:
            grad_detach = torch.zeros((f_real.shape[0], f_real.shape[1], 1), device=input_real.device)
        if hidden is None:
            # hidden: B, 1, C * 2
            hidden = torch.zeros((input_real.shape[0], 1, input_real.shape[-1] * 2), device=input_real.device)
        else:
            # hidden: B, 1, C * 2
            hidden = hidden.transpose(0, 1)
        # hidden_real: B, 1, C
        # hidden_imag: B, 1, C
        hidden_real, hidden_imag = hidden.chunk(2, dim=-1)

        if not self.device == torch.device('cpu'):
            # output_real: B, L, C
            # output_imag: B, L, C
            output_real, output_imag = complex_scan(
                input_real.contiguous(), input_imag.contiguous(),
                f_real.contiguous(), f_imag.contiguous(),
                hidden_real.contiguous(), hidden_imag.contiguous(), grad_detach.contiguous()
            )
            # hidden_real: B, 1, C
            hidden_real = output_real[:, -1:, :]
            # hidden_real: B, 1, C
            hidden_imag = output_imag[:, -1:, :]
        else:


            # output_real: B, L, C
            # output_imag: B, L, C
            # hidden_real: B, 1, C
            # hidden_imag: B, 1, C
            # input_real: B, L, C
            # input_imag: B, L, C
            # input_imag: B, L, C
            # f_real: B, L, C
            # f_imag: B, L, C
            output_real, output_imag, hidden_real, hidden_imag = complex_scan_cpu(
                input_real, input_imag,
                f_real, f_imag, hidden_real, hidden_imag
            )
        # hidden: B, 1, 2 * C
        hidden = torch.cat((hidden_real, hidden_imag), dim=-1)
        # hidden: 1, B, 2 * C
        hidden = hidden.transpose(0, 1)

        # output: 2, B, L, C
        output = torch.cat((output_real.unsqueeze(0), output_imag.unsqueeze(0)), dim=0)
        # output: 2, B, L, C
        output = self.middle_proj(output)
        # o: B, L, C
        # output = self.dropout(output[0] - output[1]) + o
        output = output[0] - output[1] + o
        # TODO: silu resblock
        # output = (output[0] - output[1]) * torch.nn.functional.silu(o)
        # output_last_dim_mean = output.abs().mean(dim=-1, keepdim=True)
        # output = output / (output_last_dim_mean + 1e-7)
        if self.use_ff:
            output = self.ff(output)
        return output, hidden

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
    

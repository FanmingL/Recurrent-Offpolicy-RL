import numpy as np
import torch
import torch.nn as nn
try:
    # from .scan_triton.complex_rnn_2 import complex_scan
    from .scan_triton.complex_rnn import complex_scan
    # from .scan_triton.complex_rnn_jax import complex_scan
except:
    pass
    # import traceback
    # traceback.print_exc()

from .scan_triton.complex_rnn_cpu import complex_scan_cpu
from ..ensemble_linear_model import EnsembleLinear
from ..multi_ensemble_linear_model import MultiEnsembleLinear

class EnsembleLRULayer(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            num_ensemble,
            dropout=0.0,
            batch_first=True,
            use_ff = True,
            squash_inproj=False,
    ):
        super().__init__()
        assert batch_first, f'LRU only support batch_first==True'
        self.d_model = output_dim
        self.num_ensemble = num_ensemble
        self.in_proj = MultiEnsembleLinear(input_dim, self.d_model, num_ensemble, 3, desire_ndim=4, bias=True)
        self.middle_proj = MultiEnsembleLinear(input_dim, self.d_model, num_ensemble, 2, desire_ndim=4, bias=True)
        self.dropout=torch.nn.Dropout(dropout)
        nu_log, theta_log, gamma_log = self.initializer(self.num_ensemble)
        self.params_log = nn.Parameter(torch.vstack((nu_log.unsqueeze(0), theta_log.unsqueeze(0), gamma_log.unsqueeze(0))), requires_grad=True)

        self.use_ff = use_ff
        self.squash_inproj = squash_inproj
        if self.use_ff:
            self.ff = EnsemblePositionWiseFeedForward(self.d_model, num_ensemble, dropout, desire_ndim=4)
        self.device = torch.device('cpu')

    def rnn_parameters(self):
        return self.parameters(recurse=True)

    def to(self, device):
        if not self.device == device:
            super().to(device)
            self.device = device

    def initializer(self, num_ensemble):
        # init nu, theta, gamma
        r_min, r_max = 0.9, 0.999
        u1 = torch.rand((num_ensemble, self.d_model))
        u2 = torch.rand((num_ensemble, self.d_model))
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        # self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))
        #
        # #https://arxiv.org/pdf/2303.06349.pdf Sect.3.2.2
        # r_min, r_max = 0.8, 0.99
        # u1 = np.random.random((num_ensemble, self.d_model))
        # u2 = np.random.random((num_ensemble, self.d_model))
        # nu_log = np.log(
        #     -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        # )
        # theta_log = np.log(u2 * np.pi * 2)
        # gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        #
        return nu_log, theta_log, gamma_log

    def forward(self, x, hidden: torch.Tensor=None, rnn_start=None, grad_detach=None):
        # x: [nensemble, B, L, C] or [B, L, C]
        # hidden: [1, batch_size, d_model * 2 * ensemble]
        # rnn_start: [B, L, 1]
        # u: 3, nensemble, B, L, C
        u = self.in_proj(x)
        if self.squash_inproj:
            # u: 3, nensemble, B, L, C
            u = torch.tanh(u)
        # input_real: nensemble, B, L, C
        input_real = u[0]
        # input_imag: nensemble, B, L, C
        input_imag = u[1]
        # o: nensemble, B, L, C
        o = u[2]
        # nu: nensemble, C
        # nu = torch.exp(-torch.exp(self.nu_log))
        # # theta: nensemble, C
        # theta = torch.exp(self.theta_log)
        # # gamma: nensemble, C
        # gamma = torch.exp(self.gamma_log)
        #
        # # f_real: nensemble, C
        # f_real = nu * torch.cos(theta)
        # # f_imag: nensemble, C
        # f_imag = nu * torch.sin(theta)

        params = torch.exp(self.params_log)
        nu = params[0]
        theta = params[1]
        gamma = params[2]
        lamb = torch.exp(torch.complex(-nu, theta))
        f_real = lamb.real
        f_imag = lamb.imag
        # print(f'nu: {nu.shape}, gamma: {gamma.shape}, lamb: {lamb.shape}, f_real:{f_real.shape}, f_image: {f_imag.shape}. input_real: {input_real.shape}, self.params_log: {self.params_log.shape}')

        # gamma[:, None, None, :]: nensemble, 1, 1, C
        # input_real: nensemble, B, L, C
        input_real = gamma[:, None, None, :] * input_real
        # input_imag: nensemble, B, L, C
        input_imag = gamma[:, None, None, :] * input_imag
        # f_real: nensemble, B, L, C
        f_real = f_real[:, None, None, :].expand_as(input_real)
        # f_imag: nensemble, B, L, C
        f_imag = f_imag[:, None, None, :].expand_as(input_imag)

        if rnn_start is not None:
            if len(rnn_start.shape) < len(f_real.shape):
                rnn_start = rnn_start.unsqueeze(0)
            # rnn_start: 1, B, L, C
            # f_real: nensemble, B, L, C
            f_real = f_real * (1 - rnn_start)
            # f_imag: nensemble, B, L, C
            f_imag = f_imag * (1 - rnn_start)

        # input_real: nensemble * B, L, C
        input_real = input_real.reshape((-1, input_real.shape[-2], input_real.shape[-1]))
        # input_imag: nensemble * B, L, C
        input_imag = input_imag.reshape((-1, input_imag.shape[-2], input_imag.shape[-1]))
        # f_real: nensemble * B, L, C
        f_real = f_real.reshape((-1, f_real.shape[-2], f_real.shape[-1]))
        # f_imag: nensemble * B, L, C
        f_imag = f_imag.reshape((-1, f_imag.shape[-2], f_imag.shape[-1]))

        if hidden is None:
            hidden = torch.zeros((input_real.shape[0], 1, 2 * input_real.shape[-1]), device=input_real.device)
        else:
            hidden = hidden.transpose(0, 1)
        # hidden_real, hidden_imag = hidden.chunk(2, dim=-1)
        hidden_items = hidden.chunk(2 * self.num_ensemble, dim=-1)
        hidden_items = [item.unsqueeze(0) for item in hidden_items]
        hidden_real = torch.cat(hidden_items[:self.num_ensemble], dim=0)
        hidden_imag = torch.cat(hidden_items[self.num_ensemble:], dim=0)
        # TODO: Is it right? check it !
        # although this branch will not be entered
        # hidden_real = (hidden_real_0, hidden_real_1, ..., hidden_real_num_ensemble)
        hidden_real = hidden_real.reshape((-1, 1, hidden_real.shape[-1]))
        hidden_imag = hidden_imag.reshape((-1, 1, hidden_imag.shape[-1]))

        if grad_detach is None:
            grad_detach = torch.zeros((input_real.shape[0], input_real.shape[1], 1), device=x.device)
        else:
            if grad_detach.shape[0] < input_real.shape[0]:
                grad_detach = grad_detach.unsqueeze(0).repeat_interleave(self.num_ensemble, dim=0)
                grad_detach = grad_detach.reshape((-1, grad_detach.shape[-2], grad_detach.shape[-1]))
        if not self.device == torch.device('cpu'):
            # output_real: nensemble * B, L, C
            # output_imag: nensemble * B, L, C
            output_real, output_imag = complex_scan(
                input_real.contiguous(), input_imag.contiguous(),
                f_real.contiguous(), f_imag.contiguous(),
                hidden_real.contiguous(), hidden_imag.contiguous(), grad_detach.contiguous()
            )
            # output_real: nensemble, B, L, C
            # output_imag: nensemble, B, L, C
            output_real = output_real.reshape((self.num_ensemble, -1, output_real.shape[-2], output_real.shape[-1]))
            output_imag = output_imag.reshape((self.num_ensemble, -1, output_imag.shape[-2], output_imag.shape[-1]))
            # hidden_real: nensemble, B, 1, C
            # hidden_imag: nensemble, B, 1, C
            hidden_real = output_real[..., -1:, :]
            hidden_imag = output_imag[..., -1:, :]
            hidden_real = hidden_real.reshape((self.num_ensemble, -1, 1, hidden_real.shape[-1]))
            hidden_imag = hidden_imag.reshape((self.num_ensemble, -1, 1, hidden_imag.shape[-1]))
        else:

            output_real, output_imag, hidden_real, hidden_imag = complex_scan_cpu(
                input_real, input_imag,
                f_real, f_imag, hidden_real, hidden_imag
            )
            output_real = output_real.reshape((self.num_ensemble, -1, output_real.shape[-2], output_real.shape[-1]))
            output_imag = output_imag.reshape((self.num_ensemble, -1, output_imag.shape[-2], output_imag.shape[-1]))
            hidden_real = hidden_real.reshape((self.num_ensemble, -1, 1, hidden_real.shape[-1]))
            hidden_imag = hidden_imag.reshape((self.num_ensemble, -1, 1, hidden_imag.shape[-1]))
        # hidden_real: B, 1, C * nensemble
        # hidden_imag: B, 1, C * nensemble
        hidden_real = torch.cat([hidden_real[i] for i in range(hidden_real.shape[0])], dim=-1)
        hidden_imag = torch.cat([hidden_imag[i] for i in range(hidden_imag.shape[0])], dim=-1)
        # hidden_real = hidden_real.reshape((1, 1, hidden_real.shape[-1] * hidden_real.shape[0]))
        # hidden_imag = hidden_imag.reshape((1, 1, hidden_imag.shape[-1] * hidden_imag.shape[0]))
        # hidden: B, 1, C * nensemble * 2
        hidden = torch.cat((hidden_real, hidden_imag), dim=-1)
        # hidden: 1, B, C * nensemble * 2
        hidden = hidden.transpose(0, 1)

        # output: 2, nensemble, B, L, C
        output = torch.cat((output_real.unsqueeze(0), output_imag.unsqueeze(0)), dim=0)
        # output: 2, nensemble, B, L, C
        output = self.middle_proj(output)
        # output: nensemble, B, L, C
        output = output[0] - output[1] + o
        # TODO: silu resblock
        # output = (output[0] - output[1]) * torch.nn.functional.silu(o)
        # output = self.dropout(output[0] - output[1]) + o
        # output_last_dim_mean = output.abs().mean(dim=-1, keepdim=True)
        # print(f'last dim max: {output_last_dim_mean.max()}, {output.abs().max()}, {f_real.abs().max()}, {f_imag.abs().max()}')
        # output = output / (output_last_dim_mean + 1e-7)
        if self.use_ff:
            output = self.ff(output)
        return output, hidden


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

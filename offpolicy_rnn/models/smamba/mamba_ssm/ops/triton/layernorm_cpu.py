import math

import torch
import torch.nn.functional as F

def layer_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False):
    dtype = x.dtype
    if residual_in_fp32:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if residual_in_fp32:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )
    return out if not prenorm else (out, x)


def rms_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False):
    dtype = x.dtype
    if residual_in_fp32:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if residual_in_fp32:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


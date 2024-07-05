# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial

from einops import rearrange, repeat
try:
    from .mamba_ssm.ops.selective_scan_interface_new import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn = None
    mamba_inner_fn = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from .mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
from .mamba_ssm.ops.triton.layernorm_cpu import RMSNorm as RMSNormCPU
from .mamba_ssm.ops.triton.layernorm_cpu import layer_norm_fn as layer_norm_fn_cpu
from .mamba_ssm.ops.triton.layernorm_cpu import rms_norm_fn as rms_norm_fn_cpu
try:
    from .mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = RMSNormCPU, layer_norm_fn_cpu, rms_norm_fn_cpu


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        # use_ff=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv_hidden_dim = self.d_model * self.expand * self.d_conv
        self.ssm_hidden_dim = self.d_model * self.expand * self.d_state
        self.desired_hidden_dim = self.conv_hidden_dim + self.ssm_hidden_dim
        self.use_conv = self.d_conv > 0
        if self.use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # self.use_ff = use_ff
        # if use_ff:
        #     self.ff = PositionWiseFeedForward(d_model, 0.0)

    def forward(self, x, hidden=None, rnn_start=None, mask=None):
        if x.device == torch.device('cpu') or x.shape[-2] == 1:
            if hidden is None:
                conv_state, ssm_state = self.allocate_inference_cache(x.shape[0], max_seqlen=2048)
            else:
                if self.use_conv:
                    conv_state = hidden[0, :, :self.conv_hidden_dim].reshape(
                        (x.shape[0], self.d_model * self.expand, self.d_conv))
                else:
                    conv_state = None
                ssm_state = hidden[0, :, self.conv_hidden_dim:].reshape(
                    (x.shape[0], self.d_model * self.expand, self.d_state))
            out_list = []
            for i in range(x.shape[-2]):
                out, conv_state, ssm_state = self.step(x[..., i:i+1, :], conv_state, ssm_state)
                out_list.append(out)
            if len(out_list) == 1:
                out = out_list[0]
            else:
                out = torch.cat(out_list, dim=-2)
            if self.use_conv:
                hidden = torch.cat((
                    conv_state.reshape((1, x.shape[0], -1)),
                    ssm_state.reshape((1, x.shape[0], -1)),
                ), dim=-1)
            else:
                hidden = ssm_state.reshape((1, x.shape[0], -1))
        else:
            out = self.forward_sequential(x, mask, rnn_start)
            if hidden is None:
                hidden = torch.zeros((1, x.shape[0], self.conv_hidden_dim + self.ssm_hidden_dim), device=x.device)
        return out, hidden

    def forward_sequential(self, hidden_states, mask=None, start=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if mask is not None:
            mask = mask.transpose(-2, -1).contiguous()
        if start is not None:
            start = start.repeat_interleave(self.d_inner, dim=-1).transpose(-2, -1).contiguous()
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and self.d_conv <= 4:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight if self.use_conv else None,
                self.conv1d.bias if self.use_conv else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                start,
                mask,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if mask is not None:
                x = mask * x
            x = self.act(self.conv1d(x)[..., :seqlen])
            # if causal_conv1d_fn is None:
            #     if mask is not None:
            #         x = mask * x
            #     x = self.act(self.conv1d(x)[..., :seqlen])
            # else:
            #     assert self.activation in ["silu", "swish"]
            #     if mask is not None:
            #         x = mask * x
            #     x = causal_conv1d_fn(
            #         x=x,
            #         weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            #         bias=self.conv1d.bias,
            #         activation=self.activation,
            #     )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                start,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        # if self.use_ff:
        #     out = self.ff(out)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)
        if self.use_conv:
            # Conv step
            if causal_conv1d_update is None or hidden_states.device == torch.device('cpu') or self.d_conv > 4:
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)  # Update state (B D W)
                conv_state = torch.cat((conv_state[:, :, :-1], x.unsqueeze(-1)), dim=-1)
                # conv_state[:, :, -1] = x
                x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
                if self.conv1d.bias is not None:
                    x = x + self.conv1d.bias
                x = self.act(x).to(dtype=dtype)
            else:
                x = causal_conv1d_update(
                    x,
                    conv_state,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None or hidden_states.device == torch.device('cpu'):
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state = ssm_state * dA + rearrange(x, "b d -> b d 1") * dB
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        # if self.use_ff:
        #     out = self.ff(out)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        if self.use_conv:
            conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
            conv_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
            )
        else:
            conv_state = None
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer: Mamba = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm, RMSNormCPU)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, hidden=None, rnn_start=None, mask=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            if hidden_states.device == torch.device('cpu'):
                fused_add_norm_fn = rms_norm_fn_cpu if isinstance(self.norm, (RMSNorm, RMSNormCPU)) else layer_norm_fn_cpu
            else:
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, (RMSNorm, RMSNormCPU)) else layer_norm_fn
            # fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, (RMSNorm, RMSNormCPU)) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        out, hidden = self.mixer(hidden_states, hidden, rnn_start, mask)
        return out, hidden, residual

    # def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #     return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class BlockList(torch.nn.Module):
    def __init__(self, block_num, dim, d_conv=4, d_state=16, fused_add_norm=True,
                 rms_norm=True, residual_in_fp32=True, use_ff=False):
        super().__init__()
        self.block_num = block_num
        self.fused_add_norm = fused_add_norm
        self.rms_norm = rms_norm
        self.norm_epsilon = 1e-8
        self.d_conv = d_conv
        self.residual_in_fp32 = residual_in_fp32
        self.layers = nn.ModuleList(
            [
                self.create_block(
                    dim,
                    ssm_cfg={
                        "d_conv": d_conv,
                        "d_state": d_state
                    },
                    norm_epsilon=self.norm_epsilon,
                    rms_norm=self.rms_norm,
                    residual_in_fp32=self.residual_in_fp32,
                    fused_add_norm=self.fused_add_norm,
                    layer_idx=i,
                    # **factory_kwargs,
                )
                for i in range(self.block_num)
            ]
        )
        self.desired_hidden_dim = self.layers[0].mixer.desired_hidden_dim * self.block_num
        self.use_ff = use_ff
        if use_ff:
            self.head = PositionWiseFeedForward(d_model=dim, dropout=0.0, eps=self.norm_epsilon)
        else:
            self.head = nn.Linear(dim, dim, bias=False)
            self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
                dim, eps=self.norm_epsilon  # , **factory_kwargs
            )

        self.apply(
            partial(
                _init_weights,
                n_layer=self.block_num)
            )

    def create_block(
            self,
            d_model,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=False,
            residual_in_fp32=False,
            fused_add_norm=False,
            layer_idx=None,
            device=None,
            dtype=None,
    ) -> Block:
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )

        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
        return block

    def forward(self, x, hidden=None, rnn_start=None, mask=None):
        if hidden is None:
            hidden = torch.zeros((x.shape[0], 1, self.desired_hidden_dim), device=x.device)
        hiddens = torch.chunk(hidden, self.block_num, dim=-1)
        residual = None
        hidden_output = []
        for i in range(self.block_num):
            block: Block = self.layers[i]
            x, hidden, residual = block.forward(x, residual, hiddens[i], rnn_start, mask)
            hidden_output.append(hidden)
        # I find the following is not required FF
        if not self.use_ff:
            if not self.fused_add_norm:
                residual = (x + residual) if residual is not None else x
                x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                if x.device == torch.device('cpu'):
                    fused_add_norm_fn = rms_norm_fn_cpu if isinstance(self.norm_f, (RMSNorm, RMSNormCPU)) else layer_norm_fn_cpu
                else:
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, (RMSNorm, RMSNormCPU)) else layer_norm_fn
                x = fused_add_norm_fn(
                    x,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
        else:
            x = (x + residual) if residual is not None else x
        x = self.head(x)
        hidden = torch.cat(hidden_output, dim=-1)
        return x, hidden

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.0, eps=1e-5):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)

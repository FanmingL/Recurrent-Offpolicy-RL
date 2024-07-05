import math

import numpy as np
import torch
from torch import nn
from flash_attn.modules.mha import MHA
from flash_attn.bert_padding import unpad_input_for_concatenated_sequences, pad_input
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None
    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

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


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x


def pre_norm(module, x, norm, dropout, *args, **kwargs):
    return dropout(module(norm(x), *args, **kwargs)) + x


def post_norm(module, x, norm, dropout, *args, **kwargs):
    return norm(dropout(module(x, *args, **kwargs)) + x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, layer_idx=None, ln=True):
        super(DecoderLayer, self).__init__()
        self.mha = MHA(embed_dim=d_model, num_heads=nhead,
                       dropout=dropout, causal=True, layer_idx=layer_idx,
                       use_alibi=True, use_flash_attn=True, return_residual=False,
                       fused_bias_fc=True, dtype=torch.bfloat16)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(d_model) if ln else RMSNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model) if ln else RMSNorm(d_model)

    def forward(self, x, inference_params=None, **kwargs):

        x_residual = x
        x = self.mha_norm(x)
        original_x_dtype = x.dtype

        x = x.to(torch.bfloat16)
        # with torch.cuda.amp.autocast():
        x = self.mha.forward(x, inference_params=inference_params, **kwargs)
        x = x.to(original_x_dtype)
        x = self.dropout(x) + x_residual
        x = pre_norm(self.ffn, x, self.ffn_norm, self.dropout)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_layer, dropout=0.1, ln=True):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.n_layer = n_layer
        self.d_model = d_model
        self.decoder_layers = torch.nn.ModuleList([
            DecoderLayer(
                d_model, n_head, d_ff, dropout=dropout, layer_idx=i, ln=ln
            ) for i in range(n_layer)
        ])
        self.output_ln = nn.LayerNorm(d_model) if ln else RMSNorm(d_model)
        self.output_fc = torch.nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None, seqlens=None):
        if seqlens is not None:
            batch, seqlen = x.shape[:2]
            x, indices, cu_seqlens, max_seqlen_in_batch = unpad_input_for_concatenated_sequences(x, seqlens)
            max_seqlen_in_batch = int(max_seqlen_in_batch)
            seq_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen_in_batch,
            }
        else:
            seq_kwargs = {}
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, inference_params, **seq_kwargs)
        x = self.output_ln(x)
        x = self.output_fc(x)
        if seqlens is not None:
            x = pad_input(x, indices, batch, seqlen)
        return x

def main_onestep():
    B, L, C = 5, 512, 1024
    n_num = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((B, L, C)).to(device)
    transformer_decoder = TransformerDecoder(C, n_num, 4 * C, 10, dropout=0.0)
    transformer_decoder.train()
    transformer_decoder.to(device)
    y = transformer_decoder(x, None)
    inference_params = InferenceParams(max_seqlen=1024, max_batch_size=B)
    interval = 1

    for i in range(0, L, interval):
        xi = x[..., i:i+interval, :]
        yi = transformer_decoder(xi, None, inference_params)
        inference_params.seqlen_offset += interval
        print(i, (yi - y[...,i:i+interval, :]).abs().max())


def main():
    B, L, C = 5, 512, 1024
    n_num = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((B, L, C)).to(device)
    transformer_decoder = TransformerDecoder(C, n_num, 4 * C, 10, dropout=0.0)
    transformer_decoder.to(device)
    transformer_decoder.eval()
    seqlens = np.zeros((B, L))
    seqlens[0, 0] = 412
    seqlens[0, 1] = 100

    seqlens[1, 0] = 312
    seqlens[1, 1] = 100
    seqlens[1, 2] = 100

    seqlens[2, 0] = 100
    seqlens[2, 1] = 100
    seqlens[2, 2] = 300
    seqlens[2, 3] = 12

    seqlens[3, 0] = 212

    seqlens[4, 0] = 512

    seqlens = torch.from_numpy(seqlens).to(device).to(torch.get_default_dtype())
    # x_reshape, indices, cu_seqlens, max_seqlen_in_batch = unpad_input_for_concatenated_sequences(x, seqlens)
    y = transformer_decoder(x, None, seqlens)
    # print(f'x shape: {x[4:5, ...].shape}')
    y_4 = transformer_decoder(x[4:5])

    y_3 = transformer_decoder(x[3:4])
    y_3_short = transformer_decoder(x[3:4, :212])
    print('full length', (y[4:5]-y_4).abs().max())
    print(f'sub length', (y[3:4]-y_3).abs().max(), '->', (y[3:4, :212]-y_3_short).abs().max())
    y_2_middle = transformer_decoder(x[2:3, 100:200])
    y_2 = transformer_decoder(x[2:3])
    print(f'middle part', (y[2:3] - y_2).abs().max(), '->', (y[2:3, 100:200] - y_2_middle).abs().max())


    # print(x_reshape.shape, indices.shape, cu_seqlens.shape, max_seqlen_in_batch)
    # print('cu_seqlens', cu_seqlens)
    print(y.shape)


if __name__ == '__main__':
    main()

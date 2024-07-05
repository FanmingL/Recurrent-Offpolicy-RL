import numpy as np
import torch

from .RNNHidden import RNNHidden
from .rnn_base import RNNBase
from typing import Optional

def slice_tensor(x, slice_num):
    assert len(x.shape) == 3, 'slice operation should be added on 3-dim tensor'
    assert x.shape[1] % slice_num == 0, f'cannot reshape length with {x.shape[1]} to {slice_num} slices'
    s = x.shape
    x = x.reshape([s[0], s[1] // slice_num, slice_num, s[2]]).transpose(0, 1)
    return x


def merge_slice_tensor(data):
    s = data.shape
    data = data.reshape(s[0] * s[1], s[2], s[3])
    return data

def multi_batch_forward(network_input: torch.Tensor, network: RNNBase, hidden: Optional[RNNHidden], require_full_rnn_output: bool=False):
    # this function does not support require full hidden is True, which could result the full_rnn_output b

    assert len(network_input.shape) >= 3, f'at least a single batch data should be input'

    seq_len = network_input.shape[-2]
    dim = network_input.shape[-1]
    batch_info = list(network_input.shape[:-2])
    if len(network_input.shape) > 3:
        # does not match the input requirement of RNN
        network_input = network_input.reshape((-1, seq_len, dim))
    if hidden is not None:
        assert hidden.hidden_batch_size == network_input.shape[0]
    output, hidden, full_rnn_output = network.meta_forward(network_input, hidden, require_full_rnn_output)
    # only output is reshaped to original shape
    output = output.reshape(tuple([*batch_info, seq_len, output.shape[-1]]))
    return output, hidden, full_rnn_output


# def _forward_fix_length_onestep(self, embedding_input: torch.Tensor, uni_model_input: torch.Tensor, hidden):
#     assert len(embedding_input.shape) >= 3, '[traj_idx, time_idx, feature_idx] is required'
#     assert embedding_input.shape[-2] == 1, f'time step should be one!'
#     batch_size = embedding_input.shape[0]
#
#     hidden_length = RNNBase.get_hidden_length(hidden)
#     if hidden_length >= self.fixed_hist_length * batch_size:
#         hidden = RNNBase.pop_hidden_state(hidden, (hidden_length - self.fixed_hist_length * batch_size + batch_size) // batch_size * batch_size)
#     if hidden is None:
#         hidden = self.make_init_state(batch_size, embedding_input.device)
#     else:
#         hidden = RNNBase.append_hidden_state(hidden,
#                                         self.make_init_state(batch_size,
#                                                                                 embedding_input.device))
#     length = RNNBase.get_hidden_length(hidden) // batch_size
#     embedding_input = torch.cat([embedding_input] * length, dim=0)
#     uni_model_input = torch.cat([uni_model_input] * length, dim=0)
#     uni_model_output, hidden, embedding_output, full_memory = self.meta_forward(embedding_input,
#                                                                                 uni_model_input,
#                                                                                 hidden)
#     return uni_model_output, hidden, embedding_output, full_memory


def fixed_length_forward_one_step(network_input: torch.Tensor, network: RNNBase, hidden: Optional[RNNHidden],
                                  fixed_length: int):
    assert network.rnn_num > 0, f'rnn num must be larger than 0!!'
    network_input_shape = list(network_input.shape)
    data_batchsize = 1 if len(network_input_shape) == 2 else np.prod(network_input_shape[:-2])
    if len(network_input_shape) == 2:
        network_input = network_input.unsqueeze(0)
    assert network_input.shape[-2] == 1, f'only one step is acceptable!!'
    repeat_num = 1

    if hidden is not None:
        if hidden.hidden_batch_size >= data_batchsize * fixed_length:
            hidden.elementwise_pop(hidden.hidden_batch_size - data_batchsize * (fixed_length - 1))
        hidden_apd = network.make_init_state(data_batchsize, hidden.device)
        hidden.elementwise_append(hidden_apd)
        repeat_num = int(hidden.hidden_batch_size // data_batchsize)
    network_input = network_input.unsqueeze(0).repeat_interleave(repeat_num, dim=0)
    output, hidden, _ = multi_batch_forward(network_input, network, hidden, require_full_rnn_output=False)
    output = output[0]
    if len(network_input_shape) == 2:
        output = output.squeeze(0)
    return output, hidden


def fixed_length_forward(network_input: torch.Tensor, network: RNNBase, hidden: Optional[RNNHidden], fixed_length: int):
    assert network.rnn_num > 0, f'rnn num must be larger than 0!!'
    network_input_shape = list(network_input.shape)
    if len(network_input_shape) == 2:
        network_input = network_input.unsqueeze(0)
    seq_len = network_input_shape[-2]
    outputs = []
    for i in range(seq_len):
        output_i, hidden = fixed_length_forward_one_step(network_input[..., i:i+1,:], network, hidden, fixed_length)
        outputs.append(output_i)
    output = torch.cat(outputs, dim=-2)
    if len(network_input_shape) == 2:
        output = output.squeeze(0)
    return output, hidden

@torch.no_grad()
def get_gradient_stats(parameters):
    grad_min = float('inf')
    grad_max = float('-inf')
    total_norm_square = 0.0

    for param in parameters:
        if param.grad is not None:
            grad_min = min(grad_min, param.grad.min().item())
            grad_max = max(grad_max, param.grad.max().item())
            param_norm = torch.sum(param.grad.data ** 2)
            total_norm_square += param_norm.item()

    return grad_min, grad_max, total_norm_square
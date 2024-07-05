import time

import torch
import torch.nn.functional as F
from torch.autograd import Function
from .pscan import pscan_complex_, pscan_complex_fix_batch_size_
import jax
import numpy as np
# @jax.vmap
# def binary_operator_diag(q_i, q_j):
#     """Binary operator for parallel scan of linear recurrence"""
#     f_real_i, f_imag_i, v_real_i, v_imag_i = q_i
#     f_real_j, f_imag_j, v_real_j, v_imag_j = q_j
#     next_f_real = f_real_i * f_real_j - f_imag_i * f_imag_j
#     next_f_imag = f_real_i * f_imag_j + f_imag_i * f_real_j
#
#     next_v_real = v_real_j + f_real_j * v_real_i - f_imag_j * v_imag_i
#     next_v_imag = v_imag_j + f_real_j * v_imag_i + f_imag_j * v_real_i
#
#
#     return next_f_real, next_f_imag, next_v_real, next_v_imag

@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

def t2j(ttensor):
    return jax.numpy.array(ttensor.cpu().detach().numpy())

def j2t(jtensor, device, dtype):
    return torch.tensor(np.array(jtensor.block_until_ready()), device=device, dtype=dtype)

def pscan_jax(f_real_clone, f_imag_clone, grad_output_real_clone, grad_output_imag_clone, reverse=False):
    dtype = f_real_clone.dtype
    device = f_real_clone.device
    f_real_clone = t2j(f_real_clone)
    f_imag_clone = t2j(f_imag_clone)
    grad_output_imag_clone = t2j(grad_output_imag_clone)
    grad_output_real_clone = t2j(grad_output_real_clone)

    Lambda_elements = f_real_clone + 1j * f_imag_clone
    Bu_elements = grad_output_real_clone + 1j * grad_output_imag_clone
    _, hidden_states = jax.lax.associative_scan(binary_operator_diag, (Lambda_elements, Bu_elements), axis=1, reverse=reverse)
    # grad_output_real_clone, grad_output_imag_clone = hidden_states
    grad_output_real_clone = hidden_states.real
    grad_output_imag_clone = hidden_states.imag
    grad_output_real_clone = j2t(grad_output_real_clone, device=device, dtype=dtype)
    grad_output_imag_clone = j2t(grad_output_imag_clone, device=device, dtype=dtype)
    return grad_output_real_clone, grad_output_imag_clone


class TritonSequentialScan_Complex(Function):
    @staticmethod
    def forward(ctx, v_real, v_imag, f_real, f_imag):
        v_real_output, v_imag_output = pscan_jax(f_real, f_imag, v_real, v_imag)
        ctx.save_for_backward(v_real, v_imag, f_real, f_imag, v_real_output, v_imag_output)
        return v_real_output, v_imag_output
            
    @staticmethod
    def backward(ctx, grad_output_real, grad_output_imag):
        # basic: grad_output_real_clone
        # grad_output_real_clone[L], grad_output_imag_clone[L] = 0
        # grad_output * \tilde{decay} where \tilde{decay} denotes conjugate (decay_imag -> -decay_imag)
        # grad_output_real_clone[t] = grad_output_real[t] + grad_output_real_clone[t+1] * decay_real[t+1] + grad_output_imag_clone[t+1] * decay_imag[t+1]
        # grad_output_imag_clone[t] = grad_output_imag[t] + grad_output_imag_clone[t+1] * decay_real[t+1] - grad_output_real_clone[t+1] * decay_imag[t+1]
        # advanced: grad_f_real
        # the one before h_real[0], h_imag[0] = 0
        # grad_f_real[t] = grad_output_real_clone[t] * h_real[t-1] + grad_output_imag_clone[t] * h_imag[t-1]
        # grad_f_imag[t] = grad_output_real_clone[t] * h_imag[t-1] - grad_output_imag_clone[t] * h_real[t-1]

        v_real, v_imag, f_real, f_imag, hidden_real, hidden_imag = ctx.saved_tensors
        f_real_clone = f_real
        f_imag_clone = f_imag
        f_real_clone[:, :-1, :].copy_(f_real_clone[:, 1:, :].clone())
        f_imag_clone[:, :-1, :].copy_(f_imag_clone[:, 1:, :].clone())
        f_real_clone[:, -1, :] = 0
        f_imag_clone[:, -1, :] = 0

        f_imag_clone = -f_imag_clone
        grad_output_real_clone, grad_output_imag_clone = pscan_jax(f_real_clone, f_imag_clone, grad_output_real, grad_output_imag, reverse=True)
        hidden_real_clone = hidden_real
        hidden_imag_clone = hidden_imag
        hidden_real_clone[:, 1:, :].copy_(hidden_real_clone[:, :-1, :].clone())
        hidden_real_clone[:, 0, :] = 0
        hidden_imag_clone[:, 1:, :].copy_(hidden_imag_clone[:, :-1, :].clone())
        hidden_imag_clone[:, 0, :] = 0

        grad_f_real = (grad_output_real_clone * hidden_real_clone + grad_output_imag_clone * hidden_imag_clone)
        grad_f_imag = (grad_output_imag_clone * hidden_real_clone - grad_output_real_clone * hidden_imag_clone)
        return grad_output_real_clone, grad_output_imag_clone, grad_f_real, grad_f_imag

complex_scan = TritonSequentialScan_Complex.apply




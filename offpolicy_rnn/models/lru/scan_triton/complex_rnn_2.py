import time

import torch
import torch.nn.functional as F
from torch.autograd import Function
from .pscan import pscan_complex_, pscan_complex_fix_batch_size_


class TritonSequentialScan_Complex(Function):
    @staticmethod
    def forward(ctx, v_real, v_imag, f_real, f_imag):
        f_real_clone = f_real.clone()
        f_imag_clone = f_imag.clone()
        v_real_clone = v_real.clone()
        v_imag_clone = v_imag.clone()
        pscan_complex_fix_batch_size_(f_real_clone, f_imag_clone, v_real_clone, v_imag_clone)
        ctx.save_for_backward(v_real, v_imag, f_real, f_imag, v_real_clone, v_imag_clone)
        return v_real_clone, v_imag_clone
            
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
        f_real_clone = f_real.clone()
        f_imag_clone = f_imag.clone()
        grad_output_real_clone = grad_output_real.clone()
        grad_output_imag_clone = grad_output_imag.clone()
        f_real_clone[:, :-1, :].copy_(f_real_clone[:, 1:, :].clone())
        f_imag_clone[:, :-1, :].copy_(f_imag_clone[:, 1:, :].clone())
        f_real_clone[:, -1, :] = 0
        f_imag_clone[:, -1, :] = 0

        f_real_clone = f_real_clone.flip(-2)
        f_imag_clone = f_imag_clone.flip(-2)
        grad_output_imag_clone = grad_output_imag_clone.flip(-2)
        grad_output_real_clone = grad_output_real_clone.flip(-2)
        f_imag_clone = -f_imag_clone
        pscan_complex_fix_batch_size_(f_real_clone, f_imag_clone, grad_output_real_clone, grad_output_imag_clone)
        grad_output_real_clone = grad_output_real_clone.flip(-2)
        grad_output_imag_clone = grad_output_imag_clone.flip(-2)

        hidden_real_clone = hidden_real.clone()
        hidden_imag_clone = hidden_imag.clone()
        hidden_real_clone[:, 1:, :].copy_(hidden_real_clone[:, :-1, :].clone())
        hidden_real_clone[:, 0, :] = 0
        hidden_imag_clone[:, 1:, :].copy_(hidden_imag_clone[:, :-1, :].clone())
        hidden_imag_clone[:, 0, :] = 0

        grad_f_real = (grad_output_real_clone * hidden_real_clone + grad_output_imag_clone * hidden_imag_clone)
        grad_f_imag = (grad_output_imag_clone * hidden_real_clone - grad_output_real_clone * hidden_imag_clone)
        return grad_output_real_clone, grad_output_imag_clone, grad_f_real, grad_f_imag

complex_scan = TritonSequentialScan_Complex.apply




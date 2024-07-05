import torch


def forward_sequential_scan_complex_no_grad(v_real, v_imag, f_real, f_imag, h_real=None, h_imag=None):
    B, L, C = v_real.shape
    # assert C % 256 == 0, 'Hidden dimension must be a multiple of 256'

    hidden_real = torch.zeros_like(v_real)
    hidden_imag = torch.zeros_like(v_imag)

    h_real = torch.zeros((B, 1, v_real.shape[-1]), device=v_real.device, dtype=v_real.dtype) if h_real is None else h_real
    h_imag = torch.zeros((B, 1, v_imag.shape[-1]), device=v_imag.device, dtype=v_imag.dtype) if h_imag is None else h_imag

    for l in range(L):
        f_real_item = f_real[:, l:l+1, :]
        f_imag_item = f_imag[:, l:l+1, :]
        h_real_new = h_real * f_real_item - h_imag * f_imag_item + v_real[:, l:l+1, :]
        h_imag_new = h_real * f_imag_item + h_imag * f_real_item + v_imag[:, l:l+1, :]

        hidden_real[:, l:l+1, :] = h_real_new
        hidden_imag[:, l:l+1, :] = h_imag_new

        h_real = h_real_new
        h_imag = h_imag_new

    return hidden_real, hidden_imag, h_real, h_imag

complex_scan_cpu = forward_sequential_scan_complex_no_grad



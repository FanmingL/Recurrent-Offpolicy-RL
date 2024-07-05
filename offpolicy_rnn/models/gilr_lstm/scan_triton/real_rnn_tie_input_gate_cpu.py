import torch


def real_scan_tie_input_gate_no_grad(v, f, h=None):
    B, L, C = v.shape
    # assert C % 256 == 0, 'Hidden dimension must be a multiple of 256'
    output = torch.zeros_like(v)
    h = torch.zeros((B, 1, v.shape[-1]), device=v.device, dtype=v.dtype) if h is None else h
    for l in range(L):
        f_item = f[:, l:l+1, :]
        h_new = h * f_item + v[:, l:l+1, :] * (1-f_item)
        output[:, l:l+1, :] = h_new
        h = h_new
    return output, h

def real_scan_tie_input_gate_no_grad_fuse(v, f, h=None):
    B, L, C = v.shape
    # assert C % 256 == 0, 'Hidden dimension must be a multiple of 256'
    output = torch.zeros_like(v)
    h = torch.zeros((B, 1, v.shape[-1]), device=v.device, dtype=v.dtype) if h is None else h
    for l in range(L):
        f_item = torch.sigmoid(f[:, l:l+1, :])
        h_new = h * f_item + v[:, l:l+1, :] * (1-f_item)
        output[:, l:l+1, :] = h_new
        h = h_new
    return output, h

scan_cpu = real_scan_tie_input_gate_no_grad
scan_cpu_fuse = real_scan_tie_input_gate_no_grad_fuse


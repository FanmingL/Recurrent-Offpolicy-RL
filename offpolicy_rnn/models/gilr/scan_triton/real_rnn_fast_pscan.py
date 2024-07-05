import time

import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch.autograd import Function
# from fastpscan.cuda_v3 import fn as pscan_cuda_fn
# from fastpscan.cuda_v4 import fn as pscan_cuda_fn
from fastpscan.triton_v2 import fn as pscan_cuda_fn


def pscan_fast(v, f):
    B, L, C = v.shape
    desire_B = max(B, 4)
    h = torch.zeros((desire_B, C), device=v.device, dtype=v.dtype)

    v = v * (1 - f)
    if L < 1024:
        v = torch.cat((v, torch.zeros((B, 1024 - L, C), device=v.device, dtype=v.dtype)), dim=-2)
        f = torch.cat((f, torch.zeros((B, 1024 - L, C), device=v.device, dtype=v.dtype)), dim=-2)
    if desire_B > B:
        v = torch.cat((v, torch.zeros((desire_B - B, 1024, C), device=v.device, dtype=v.dtype)), dim=0)
        f = torch.cat((f, torch.zeros((desire_B - B, 1024, C), device=f.device, dtype=f.dtype)), dim=0)

    Y = pscan_cuda_fn(f, v, h)
    Y = Y[:B, :L, :]
    return Y







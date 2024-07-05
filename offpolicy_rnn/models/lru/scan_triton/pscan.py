import math
import time

import torch
# import triton
# import triton.language as tl

def _step(_xa, _aa, _idx):
    # Y_{t} = Y_{t-1} x Aa_{t} + Xa_{t}
    _idx_last = _idx - 1
    _xa[..., _idx, :].add_(_aa[..., _idx, :].mul(_xa[..., _idx_last, :]))
    # Aa_{t} = Aa_{t} x Aa_{t-1}
    _aa[..., _idx, :].mul_(_aa[..., _idx_last, :])

def pscan_(A: torch.Tensor, X: torch.Tensor):
    # A : (B, L, N)
    # X : (B, L, N)

    # modifies X in place by doing a parallel scan.
    # more formally, X will be populated by these values :
    # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

    B, L, _ = A.size()
    num_steps = int(math.log2(L))

    # up sweep or reduction step
    Aa = A
    Xa = X

    for k in range(num_steps):
        T = 2 * (Xa.size(-2) // 2)

        Aa = Aa[:, :T].view(B, T // 2, 2, -1)
        Xa = Xa[:, :T].view(B, T // 2, 2, -1) #

        _step(Xa, Aa, 1)

        Aa = Aa[:, :, 1]
        Xa = Xa[:, :, 1]

    # down sweep
    for k in range(num_steps - 1, -1, -1):
        Aa = A[:, 2 ** k - 1:L:2 ** k]
        Xa = X[:, 2 ** k - 1:L:2 ** k]

        T = 2 * (Xa.size(-2) // 2)

        if T < Xa.size(-2):
            _step(Xa, Aa, -1)
            # Xa[:, -1].add_(Aa[:, -1].mul(Xa[:, -2]))
            # Aa[:, -1].mul_(Aa[:, -2])

        Aa = Aa[:, :T].view(B, T // 2, 2, -1)
        Xa = Xa[:, :T].view(B, T // 2, 2, -1)

        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
# @triton.jit
# def _complex_operator_kernel(_x_real_a_idx, _x_real_a_idx_last, _a_imag_a_idx, _a_imag_a_idx_last,
#                              _x_imag_a_idx, _x_imag_a_idx_last, _a_real_a_idx, _a_real_a_idx_last, B,
#                              BLOCK_M: tl.constexpr):
#     # Compute the thread index
#     idx = tl.program_id(0)
#     if idx < B:
#         ptr = tl.arange(0, BLOCK_M) + idx * BLOCK_M
#         x_real_a_idx = tl.load(_x_real_a_idx + ptr).to(tl.float32)
#         x_real_a_idx_last = tl.load(_x_real_a_idx_last + ptr).to(tl.float32)
#         a_imag_a_idx = tl.load(_a_imag_a_idx + ptr).to(tl.float32)
#         a_imag_a_idx_last = tl.load(_a_imag_a_idx_last + ptr).to(tl.float32)
#         x_imag_a_idx = tl.load(_x_imag_a_idx + ptr).to(tl.float32)
#         x_imag_a_idx_last = tl.load(_x_imag_a_idx_last + ptr).to(tl.float32)
#         a_real_a_idx = tl.load(_a_real_a_idx + ptr).to(tl.float32)
#         a_real_a_idx_last = tl.load(_a_real_a_idx_last + ptr).to(tl.float32)
#         x_real_a_idx = x_real_a_idx + a_real_a_idx * x_real_a_idx_last - a_imag_a_idx * x_imag_a_idx_last
#         x_imag_a_idx = x_imag_a_idx + a_real_a_idx * x_imag_a_idx_last + a_imag_a_idx * x_real_a_idx_last
#
#         tl.store(_x_real_a_idx + ptr, x_real_a_idx.to(_x_real_a_idx.dtype.element_ty))
#         tl.store(_x_imag_a_idx + ptr, x_imag_a_idx.to(_x_imag_a_idx.dtype.element_ty))
#
#         a_real_a_next = a_real_a_idx * a_real_a_idx_last - a_imag_a_idx * a_imag_a_idx_last
#         a_imag_a_next = a_imag_a_idx * a_real_a_idx_last - a_real_a_idx * a_imag_a_idx_last
#
#         tl.store(_a_real_a_idx + ptr, a_real_a_next.to(_a_real_a_idx.dtype.element_ty))
#         tl.store(_a_imag_a_idx + ptr, a_imag_a_next.to(_a_imag_a_idx.dtype.element_ty))

def _complex_operator(_x_real_a_idx, _x_real_a_idx_last, _a_imag_a_idx, _a_imag_a_idx_last,
                      _x_imag_a_idx, _x_imag_a_idx_last, _a_real_a_idx, _a_real_a_idx_last):
    # # print(_x_real_a_idx.shape)
    # if len(_x_real_a_idx.shape) == 3:
    #     _x_real_a_idx = _x_real_a_idx.squeeze(0)
    #     _x_real_a_idx_last = _x_real_a_idx_last.squeeze(0)
    #     _a_imag_a_idx = _a_imag_a_idx.squeeze(0)
    #     _a_imag_a_idx_last = _a_imag_a_idx_last.squeeze(0)
    #
    #     _x_imag_a_idx = _x_imag_a_idx.squeeze(0)
    #     _x_imag_a_idx_last = _x_imag_a_idx_last.squeeze(0)
    #     _a_real_a_idx = _a_real_a_idx.squeeze(0)
    #     _a_real_a_idx_last = _a_real_a_idx_last.squeeze(0)
    # if _x_real_a_idx.shape[0] > 0:
    #     _complex_operator_kernel[(_x_real_a_idx.shape[0],)](
    #         _x_real_a_idx.contiguous(), _x_real_a_idx_last.contiguous(), _a_imag_a_idx.contiguous(), _a_imag_a_idx_last.contiguous(),
    #         _x_imag_a_idx.contiguous(), _x_imag_a_idx_last.contiguous(), _a_real_a_idx.contiguous(), _a_real_a_idx_last.contiguous(),
    #         _x_real_a_idx.shape[0], BLOCK_M=256, num_warps=16
    #     )
    _x_real_a_idx.add_(
        _a_real_a_idx.mul(_x_real_a_idx_last) - _a_imag_a_idx.mul(_x_imag_a_idx_last)
    )
    _x_imag_a_idx.add_(
        _a_real_a_idx.mul(_x_imag_a_idx_last) + _a_imag_a_idx.mul(_x_real_a_idx_last)
    )
    _a_real_a_next = _a_real_a_idx.mul(_a_real_a_idx_last) - _a_imag_a_idx.mul(_a_imag_a_idx_last)
    _a_imag_a_next = _a_imag_a_idx.mul(_a_real_a_idx_last) + _a_real_a_idx.mul(_a_imag_a_idx_last)
    _a_real_a_idx.copy_(_a_real_a_next)
    _a_imag_a_idx.copy_(_a_imag_a_next)

def _step_complex(_x_real_a, _x_imag_a, _a_real_a, _a_imag_a, _idx):
    # Y_{t} = Y_{t-1} x Aa_{t} + Xa_{t}
    _idx_last = _idx - 1
    _x_real_a_idx = _x_real_a[..., _idx, :]
    _x_real_a_idx_last = _x_real_a[..., _idx_last, :]
    _a_imag_a_idx = _a_imag_a[..., _idx, :]
    _a_imag_a_idx_last = _a_imag_a[..., _idx_last, :]
    _x_imag_a_idx = _x_imag_a[..., _idx, :]
    _x_imag_a_idx_last = _x_imag_a[..., _idx_last, :]
    _a_real_a_idx = _a_real_a[..., _idx, :]
    _a_real_a_idx_last = _a_real_a[..., _idx_last, :]
    # print('items', X_real[0, 1, :], X_real[0, 0, :], A_real[0, 1, :], X_imag[0, 0, :], A_imag[0, 1, :])

    _complex_operator(_x_real_a_idx, _x_real_a_idx_last, _a_imag_a_idx, _a_imag_a_idx_last,
                      _x_imag_a_idx, _x_imag_a_idx_last, _a_real_a_idx, _a_real_a_idx_last)

def _step_final_complex(_x_real_a, _x_imag_a, _a_real_a, _a_imag_a):
    # Y_{t} = Y_{t-1} x Aa_{t} + Xa_{t}
    _x_real_a_idx = _x_real_a[:, 1:, 0]
    _x_real_a_idx_last = _x_real_a[:, :-1, 1]
    _a_imag_a_idx = _a_imag_a[:, 1:, 0]
    _a_imag_a_idx_last = _a_imag_a[:, :-1, 1]
    _x_imag_a_idx = _x_imag_a[:, 1:, 0]
    _x_imag_a_idx_last = _x_imag_a[:, :-1, 1]
    _a_real_a_idx = _a_real_a[:, 1:, 0]
    _a_real_a_idx_last = _a_real_a[:, :-1, 1]
    _complex_operator(_x_real_a_idx, _x_real_a_idx_last, _a_imag_a_idx, _a_imag_a_idx_last,
                      _x_imag_a_idx, _x_imag_a_idx_last, _a_real_a_idx, _a_real_a_idx_last)

def pscan_complex_(A_real: torch.Tensor, A_imag: torch.Tensor,
                   X_real: torch.Tensor, X_imag: torch.Tensor):
    # A : (B, L, N)
    # X : (B, L, N)
    # print(A_real.shape, A_imag.shape, X_real.shape, X_imag.shape)
    # modifies X in place by doing a parallel scan.
    # more formally, X will be populated by these values :
    # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
    B, L, _ = A_real.size()
    num_steps = int(math.log2(L))
    # up sweep or reduction step
    A_real_a = A_real
    A_imag_a = A_imag

    X_real_a = X_real
    X_imag_a = X_imag
    for k in range(num_steps):
        T = 2 * (X_real_a.size(-2) // 2)
        A_real_a = A_real_a[:, :T].view(B, T // 2, 2, -1) # A_real_a[:, :, 0]  A_real_a[:, :, 1]: data_{even} (0, 2, 4, ...),  data_{odd} (1, 3, 5, ...)
        A_imag_a = A_imag_a[:, :T].view(B, T // 2, 2, -1) #
        X_real_a = X_real_a[:, :T].view(B, T // 2, 2, -1)
        X_imag_a = X_imag_a[:, :T].view(B, T // 2, 2, -1) #
        _step_complex(X_real_a, X_imag_a, A_real_a, A_imag_a, 1)     # choose odd, odd <- even odd
        A_real_a = A_real_a[:, :, 1]   # preserve data_{odd} (1, 3, 5, ...)
        A_imag_a = A_imag_a[:, :, 1]   # preserve data_{odd} (1, 3, 5, ...)

        X_real_a = X_real_a[:, :, 1]   # preserve data_{odd} (1, 3, 5, ...)
        X_imag_a = X_imag_a[:, :, 1]   # preserve data_{odd} (1, 3, 5, ...)
    # down sweep
    for k in range(num_steps - 1, -1, -1):   # (num_steps - 1, num_steps - 2, ..., 0)
        A_real_a = A_real[:, 2 ** k - 1:L:2 ** k] # when k = 0, 0:L:1, full data, when k = num_steps - 1, L = 1024, num_steps = 10, 511:1024:512, 511, 1203
        A_imag_a = A_imag[:, 2 ** k - 1:L:2 ** k]
        X_real_a = X_real[:, 2 ** k - 1:L:2 ** k]
        X_imag_a = X_imag[:, 2 ** k - 1:L:2 ** k]

        T = 2 * (A_real_a.size(-2) // 2)

        if T < A_real_a.size(-2):
            _step_complex(X_real_a, X_imag_a, A_real_a, A_imag_a, -1) # choose odd, odd <- odd, even
        A_real_a = A_real_a[:, :T].view(B, T // 2, 2, -1)
        A_imag_a = A_imag_a[:, :T].view(B, T // 2, 2, -1)
        X_real_a = X_real_a[:, :T].view(B, T // 2, 2, -1)
        X_imag_a = X_imag_a[:, :T].view(B, T // 2, 2, -1)
        _step_final_complex(X_real_a, X_imag_a, A_real_a, A_imag_a)

try:
    import triton
    import triton.language as tl


    @triton.jit
    def _complex_operator_fix_batch_tl_(_x_real_a, _x_real_a_last, _a_imag_a, _a_imag_a_last,
                                        _x_imag_a, _x_imag_a_last, _a_real_a, _a_real_a_last, mask, B,
                                        BLOCK_M: tl.constexpr):
        # Compute the thread index
        idx = tl.program_id(0)
        if idx < B:
            ptr = tl.arange(0, BLOCK_M) + idx * BLOCK_M
            x_real_a = tl.load(_x_real_a + ptr).to(tl.float32)
            x_real_a_last = tl.load(_x_real_a_last + ptr).to(tl.float32)
            a_imag_a = tl.load(_a_imag_a + ptr).to(tl.float32)
            a_imag_a_last = tl.load(_a_imag_a_last + ptr).to(tl.float32)
            x_imag_a = tl.load(_x_imag_a + ptr).to(tl.float32)
            x_imag_a_last = tl.load(_x_imag_a_last + ptr).to(tl.float32)
            a_real_a = tl.load(_a_real_a + ptr).to(tl.float32)
            a_real_a_last = tl.load(_a_real_a_last + ptr).to(tl.float32)
            mask_a = tl.load(mask + ptr).to(tl.float32)

            x_real_a = (x_real_a + a_real_a * x_real_a_last - a_imag_a * x_imag_a_last) * mask_a
            x_imag_a = (x_imag_a + a_real_a * x_imag_a_last + a_imag_a * x_real_a_last) * mask_a

            tl.store(_x_real_a + ptr, x_real_a.to(_x_real_a.dtype.element_ty))
            tl.store(_x_imag_a + ptr, x_imag_a.to(_x_imag_a.dtype.element_ty))

            a_real_a_next = (a_real_a * a_real_a_last - a_imag_a * a_imag_a_last) * mask_a + (1 - mask_a) * a_real_a
            a_imag_a_next = (a_imag_a * a_real_a_last - a_real_a * a_imag_a_last) * mask_a + (1 - mask_a) * a_imag_a

            tl.store(_a_real_a + ptr, a_real_a_next.to(_a_real_a.dtype.element_ty))
            tl.store(_a_imag_a + ptr, a_imag_a_next.to(_a_imag_a.dtype.element_ty))
except:
    pass

def _complex_operator_fix_batch_tl(_x_real_a: torch.Tensor, _x_real_a_last: torch.Tensor, _a_imag_a: torch.Tensor, _a_imag_a_last: torch.Tensor,
                      _x_imag_a: torch.Tensor, _x_imag_a_last: torch.Tensor, _a_real_a: torch.Tensor, _a_real_a_last: torch.Tensor, mask: torch.Tensor):
    B = _x_real_a.shape[0]
    _complex_operator_fix_batch_tl_[(B,)](
        _x_real_a, _x_real_a_last, _a_imag_a, _a_imag_a_last,
        _x_imag_a, _x_imag_a_last, _a_real_a, _a_real_a_last, mask, B, 256, num_warps=8
    )
    pass


@torch.jit.script
def _complex_operator_fix_batch(_x_real_a: torch.Tensor, _x_real_a_last: torch.Tensor, _a_imag_a: torch.Tensor, _a_imag_a_last: torch.Tensor,
                      _x_imag_a: torch.Tensor, _x_imag_a_last: torch.Tensor, _a_real_a: torch.Tensor, _a_real_a_last: torch.Tensor, mask: torch.Tensor):
    not_mask = 1 - mask

    _x_real_a.add_(
        (_a_real_a * _x_real_a_last - _a_imag_a * _x_imag_a_last) * mask
    )
    _x_imag_a.add_(
        (_a_real_a * _x_imag_a_last + _a_imag_a * _x_real_a_last) * mask
    )
    _a_real_a_next = _a_real_a.mul(_a_real_a_last) - _a_imag_a.mul(_a_imag_a_last)
    _a_imag_a_next = _a_imag_a.mul(_a_real_a_last) + _a_real_a.mul(_a_imag_a_last)
    _a_real_a.copy_(not_mask * _a_real_a + mask * _a_real_a_next)
    _a_imag_a.copy_(not_mask * _a_imag_a + mask * _a_imag_a_next)


def pscan_complex_fix_batch_size_(A_real: torch.Tensor, A_imag: torch.Tensor,
                                  X_real: torch.Tensor, X_imag: torch.Tensor):
    # A : (B, L, N)
    # X : (B, L, N)
    # print(A_real.shape, A_imag.shape, X_real.shape, X_imag.shape)
    # modifies X in place by doing a parallel scan.
    # more formally, X will be populated by these values :
    # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
    B, L, D = A_real.size()
    num_steps = int(math.log2(L))
    # up sweep or reduction step
    A_real_out = A_real
    A_imag_out = A_imag
    X_real_out = X_real
    X_imag_out = X_imag
    dtype = A_real.dtype
    A_real_last = torch.cat((torch.zeros(B, L, D, device=A_real.device, dtype=dtype),
                             A_real[:, :, :].to(dtype)), dim=-2)
    A_imag_last = torch.cat((torch.zeros(B, L, D, device=A_imag.device, dtype=dtype),
                             A_imag[:, :, :].to(dtype)), dim=-2)
    X_real_last = torch.cat((torch.zeros(B, L, D, device=X_real.device, dtype=dtype),
                             X_real[:, :, :].to(dtype)), dim=-2)
    X_imag_last = torch.cat((torch.zeros(B, L, D, device=X_imag.device, dtype=dtype),
                             X_imag[:, :, :].to(dtype)), dim=-2)
    A_real = A_real_last[:, L:, :]
    A_imag = A_imag_last[:, L:, :]
    X_real = X_real_last[:, L:, :]
    X_imag = X_imag_last[:, L:, :]

    # data = torch.cat([torch.arange(0, L).unsqueeze(-1).unsqueeze(0) + i * 10000 for i in range(B)])
    mask = torch.ones((B, L, D), device=A_real.device, dtype=dtype)
    mask_a = mask
    for k in range(num_steps):
        # k = 0
        # full step: 0, 1, 2, 3, 4, ...
        # executed odd: 1, 3, 5, 7,
        # last even: 0, 2, 4, 6, ...
        # preserved 1, 3, 5, 7
        # k = 1
        # full step: 1, 3, 5, 7
        # executed odd: 3, 7, 11, 15, 19
        # last even: 1, 5, 9, 13
        # preserved: 3, 7, 11, 15, 19
        # k = 2
        # full step: 3, 7, 11, 15, 19
        # executed odd: 7, 15, 23
        # last even: 3, 11, 19
        # preserved: 7, 15, 23
        # k
        # T0 = 2 * (L // 2)
        # Tk = 2 * (T_{k-1} // 2)

        # iterative law
        # Tk = 2 * (L // (2 ** (k+1)))
        # full step: start: (2**k)-1; interval: 2**k; end: Tk * (2**k) + (2**k)
        # executed odd: start: 2**(k+1) - 1, interval: 2**(k+1); end: Tk * (2**(k+1)) + 2**(k+1)
        # last even: start: (2**k)-1, interval: 2**(k+1); end: Tk * (2**(k+1)) + (2**k)
        # preserved: start: 2**(k+1) - 1, interval: 2**(k+1); end: Tk * (2**(k+1)) + 2**(k+1)

        T = 2 * (mask_a.size(-2) // 2)
        A_real_last_k = A_real_last[:, L - 2**k:2 * L - 2**k, :]
        A_imag_last_k = A_imag_last[:, L - 2**k:2 * L - 2**k, :]
        X_real_last_k = X_real_last[:, L - 2**k:2 * L - 2**k, :]
        X_imag_last_k = X_imag_last[:, L - 2**k:2 * L - 2**k, :]

        mask_a[:, T:] = 0
        mask_a = mask_a[:, :T].view(B, T // 2, 2, -1)  #  # A_real_a[:, :, 0]  A_real_a[:, :, 1]: data_{even} (0, 2, 4, ...),  data_{odd} (1, 3, 5, ...)
        # 3, 7, 11, 15, 19; 3, 11, 19; 7, 15
        mask_a[:, :, 0] = 0
        mask_a = mask_a[:, :, 1]
        _complex_operator_fix_batch(X_real, X_real_last_k,
                                    A_imag, A_imag_last_k,
                                    X_imag, X_imag_last_k,
                                    A_real, A_real_last_k, mask)  # choose odd, odd <- even odd


    for k in range(num_steps - 1, -1, -1):  # (num_steps - 1, num_steps - 2, ..., 0)

        # full step: start: 2 ** k - 1; interval: 2 ** k; end: L
        # Tk = L // 2**k
        # final_idx: (L // 2**k - 1) * 2**k + 2 ** k - 1 = (L // 2 **k) * (2 ** k) - 1
        #
        # T = 2 * (L // 2**(k+1))
        # interval = 2 ** k
        # last_step_num: 2 ** k

        # if T < Tk: # Tk is odd, Tk // 2 = (Tk - 1) // 2
        #      T = (Tk - 1)
        #      current index: (L // 2 **k) * (2 ** k) - 1
        #      last index: (L // 2 **k) * (2 ** k) - 1 - 2 ** k

        # executed: 3 * (2 ** k) - 1; interval: 2 ** (k+1); end: 2 * (L // 2**(k+1)) * (2 ** k)
        # interval = 2 ** k

        mask[:] = 0
        mask_a = mask[:, 2 ** k - 1:L:2 ** k]

        T = 2 * (mask_a.size(-2) // 2)
        interval = 2 ** k
        A_real_last_k = A_real_last[:, L - interval:2 * L - interval, :]
        A_imag_last_k = A_imag_last[:, L - interval:2 * L - interval, :]
        X_real_last_k = X_real_last[:, L - interval:2 * L - interval, :]
        X_imag_last_k = X_imag_last[:, L - interval:2 * L - interval, :]
        if T < mask_a.size(-2):

            mask_a[:, -1, :] = 1
            _complex_operator_fix_batch(X_real, X_real_last_k,
                                        A_imag, A_imag_last_k,
                                        X_imag, X_imag_last_k,
                                        A_real, A_real_last_k, mask)  # choose odd, odd <- even odd\
        mask_a[:, T:] = 0
        mask_a = mask_a[:, :T].view(B, T // 2, 2, D)
        mask_a[:, 1:, 0] = 1

        _complex_operator_fix_batch(X_real, X_real_last_k,
                                    A_imag, A_imag_last_k,
                                    X_imag, X_imag_last_k,
                                    A_real, A_real_last_k, mask)  # choose odd, odd <- even odd
    A_real_out.copy_(A_real)
    A_imag_out.copy_(A_imag)
    X_real_out.copy_(X_real)
    X_imag_out.copy_(X_imag)


@torch.jit.script
def pscan_complex_fix_batch_size_jit(A_real: torch.Tensor, A_imag: torch.Tensor, X_real: torch.Tensor, X_imag: torch.Tensor):
    # A : (B, L, N)
    # X : (B, L, N)
    # print(A_real.shape, A_imag.shape, X_real.shape, X_imag.shape)
    # modifies X in place by doing a parallel scan.
    # more formally, X will be populated by these values :
    # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
    fix_batch_time_list = []
    # over_start_time = time.time()
    B, L, D = A_real.size()
    num_steps = int(torch.log2(torch.ones((1,), device=A_real.device) * L))
    # up sweep or reduction step
    A_real_out = A_real
    A_imag_out = A_imag
    X_real_out = X_real
    X_imag_out = X_imag

    A_real_last = torch.cat((torch.zeros(B, L, D, device=A_real.device, dtype=A_real.dtype),
                             A_real[:, :, :]), dim=-2)
    A_imag_last = torch.cat((torch.zeros(B, L, D, device=A_imag.device, dtype=A_imag.dtype),
                             A_imag[:, :, :]), dim=-2)
    X_real_last = torch.cat((torch.zeros(B, L, D, device=X_real.device, dtype=X_real.dtype),
                             X_real[:, :, :]), dim=-2)
    X_imag_last = torch.cat((torch.zeros(B, L, D, device=X_imag.device, dtype=X_imag.dtype),
                             X_imag[:, :, :]), dim=-2)
    A_real = A_real_last[:, L:, :]
    A_imag = A_imag_last[:, L:, :]
    X_real = X_real_last[:, L:, :]
    X_imag = X_imag_last[:, L:, :]

    mask = torch.ones((B, L, D), device=A_real.device, dtype=A_real.dtype)
    mask_a = mask
    for k in range(num_steps):
        T = 2 * (mask_a.size(-2) // 2)
        indicies = torch.arange(L - 2 ** k, 2 * L - 2 ** k, device=A_real.device).long()
        A_real_last_k = A_real_last[:, indicies, :]
        A_imag_last_k = A_imag_last[:, indicies, :]
        X_real_last_k = X_real_last[:, indicies, :]
        X_imag_last_k = X_imag_last[:, indicies, :]

        mask_a[:, T:] = 0
        mask_a = mask_a[:, :T].view(B, T // 2, 2, -1)  #  # A_real_a[:, :, 0]  A_real_a[:, :, 1]: data_{even} (0, 2, 4, ...),  data_{odd} (1, 3, 5, ...)
        # 3, 7, 11, 15, 19; 3, 11, 19; 7, 15
        mask_a[:, :, 0] = 0
        mask_a = mask_a[:, :, 1]
        _complex_operator_fix_batch(X_real, X_real_last_k,
                                    A_imag, A_imag_last_k,
                                    X_imag, X_imag_last_k,
                                    A_real, A_real_last_k, mask)  # choose odd, odd <- even odd


    for k in range(num_steps - 1, -1, -1):  # (num_steps - 1, num_steps - 2, ..., 0)
        mask[:] = 0
        T = 2 * (mask_a.size(-2) // 2)
        interval = 2 ** k
        indicies_mask = torch.arange(interval - 1, L, interval, device=A_real.device).long()
        mask_a = mask[:, indicies_mask]

        indicies = torch.arange(L - interval, 2 * L - interval, device=A_real.device).long()
        A_real_last_k = A_real_last[:, indicies, :]
        A_imag_last_k = A_imag_last[:, indicies, :]
        X_real_last_k = X_real_last[:, indicies, :]
        X_imag_last_k = X_imag_last[:, indicies, :]
        if T < mask_a.size(-2):

            mask_a[:, -1, :] = 1
            _complex_operator_fix_batch(X_real, X_real_last_k,
                                        A_imag, A_imag_last_k,
                                        X_imag, X_imag_last_k,
                                        A_real, A_real_last_k, mask)  # choose odd, odd <- even odd\

        mask_a[:, T:] = 0
        mask_a = mask_a[:, :T].view(B, T // 2, 2, D)
        mask_a[:, 1:, 0] = 1
        _complex_operator_fix_batch(X_real, X_real_last_k,
                                    A_imag, A_imag_last_k,
                                    X_imag, X_imag_last_k,
                                    A_real, A_real_last_k, mask)  # choose odd, odd <- even odd

    A_real_out.copy_(A_real)
    A_imag_out.copy_(A_imag)
    X_real_out.copy_(X_real)
    X_imag_out.copy_(X_imag)

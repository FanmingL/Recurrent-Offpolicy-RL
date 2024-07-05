import torch
import triton
import triton.language as tl
from einops import einsum, rearrange, repeat
from numpy import dtype

assert triton.__version__ != '2.1.0', 'Triton 2.1.0 is missing enable_fp_fusion. Triton 2.2.0 is required for numerical stability of this implementation.'

inv_ln2 = 1.44269504


# credit: https://github.com/proger/accelerated-scan/blob/b9edbad65c673f9a1915efe51dc6bbf50fd7f8c4/accelerated_scan/triton.py

@torch.jit.script
def reduce(H, C):
    return (H * C.unsqueeze(-2)).sum(-1)


@triton.jit
def fwd_recurrence(
        A,
        B,
        C,
        Dt,
        X,
        Y,
        H,
        start,
        initial_state,
        T: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)

    dt_ptr = Dt + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    u_ptr = X + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    o_ptr = Y + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    start_ptr = start + i_bh * T
    h = tl.zeros([BV, K], dtype=tl.float32)

    b_ptr = B + i_bh * T * K + tl.arange(0, K)

    A = A + ((i_v * BV) + tl.arange(0, BV)
    [:, None]) * K + tl.arange(0, K)[None, :]
    _A = tl.load(A)

    H_ptr = H + i_bh * T * D * K + \
            (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :]

    h += tl.load(initial_state + i_bh * D * K + (i_v * BV +
                                                 tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :])

    for i in range(T):
        b = tl.load(b_ptr).to(tl.float32)
        dt = tl.load(dt_ptr)
        start_flag = tl.load(start_ptr).to(tl.float32)
        u = tl.load(u_ptr)
        x_dt = u * dt
        x_dt_b = x_dt[:, None] * b[None, :]
        dt_a = tl.exp(dt[:, None] * _A) * (1 - start_flag)
        h = h * dt_a + x_dt_b
        tl.store(H_ptr, h)

        b_ptr += K
        dt_ptr += D
        start_ptr += 1
        u_ptr += D
        o_ptr += D
        H_ptr += D * K


@triton.jit
def bwd_recurrence(
        A,
        B,
        C,
        U,
        Dt,
        DO,
        H,
        start,
        DA,
        DB,
        DC,
        dDt,
        dU,
        batch,
        initial_state,
        grad_detach,
        T: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    NV = tl.cdiv(D, BV)

    dt_ptr = Dt + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    ddt_ptr = dDt + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    u_ptr = U + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    du_ptr = dU + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    do_ptr = DO + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D

    start_ptr = start + i_bh * T + (T-1)
    grad_detach_ptr = grad_detach + i_bh * T + (T-1)

    dh = tl.zeros([BV, K], dtype=tl.float32)
    dA = tl.zeros([BV, K], dtype=tl.float32)

    b_ptr = B + i_bh * T * K + tl.arange(0, K) + (T - 1) * K
    c_ptr = C + i_bh * T * K + tl.arange(0, K) + (T - 1) * K
    dc_ptr = DC + (i_bh + batch * i_v) * T * K + tl.arange(0, K) + (T - 1) * K
    db_ptr = DB + (i_bh + batch * i_v) * T * K + tl.arange(0, K) + (T - 1) * K

    A = A + ((i_v * BV) + tl.arange(0, BV)
    [:, None]) * K + tl.arange(0, K)[None, :]
    _A = tl.load(A)
    H_ptr = H + i_bh * T * D * K + \
            (i_v * BV + tl.arange(0, BV)[:, None]) * K + \
            tl.arange(0, K)[None, :] + (T - 1) * D * K

    for i in range(T):
        h = tl.load(H_ptr)
        if i < T - 1:
            next_h = tl.load(H_ptr - D * K)
        else:
            next_h = tl.load(
                initial_state + i_bh * D * K + (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :])
        b = tl.load(b_ptr).to(tl.float32)
        c = tl.load(c_ptr).to(tl.float32)
        do = tl.load(do_ptr).to(tl.float32)
        u = tl.load(u_ptr).to(tl.float32)
        dt = tl.load(dt_ptr).to(tl.float32)
        start_flag = tl.load(start_ptr).to(tl.float32)
        grad_detach_flag = tl.load(grad_detach_ptr).to(tl.float32)
        # detach grad here
        dh = dh * (1 - grad_detach_flag)
        # dA = dA * (1 - grad_detach_flag)
        # gradient wrt output proj
        dc = tl.sum(h * do[:, None], axis=0)
        tl.store(dc_ptr, dc)

        # graident wrt input
        dh += do[:, None] * c[None, :]
        dt_u = dt * u
        db = tl.sum(dh * dt_u[:, None], axis=0)
        tl.store(db_ptr, db)
        ddt_u = tl.sum(dh * b[None, :], axis=1)
        ddt = ddt_u * u
        du = ddt_u * dt
        tl.store(du_ptr, du)

        # gradient wrt decay
        dt_a = tl.exp(dt[:, None] * _A) * (1 - start_flag)
        dh *= dt_a

        d_decay = dh * next_h
        dA += d_decay * dt[:, None]
        ddt += tl.sum(d_decay * _A, axis=1)
        tl.store(ddt_ptr, ddt)

        # update ptr
        b_ptr -= K
        c_ptr -= K
        dc_ptr -= K
        db_ptr -= K
        dt_ptr -= D
        ddt_ptr -= D
        u_ptr -= D
        du_ptr -= D
        do_ptr -= D
        H_ptr -= D * K
        start_ptr -= 1
        grad_detach_ptr -= 1

    DA_ptr = DA + i_bh * D * K + \
             (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :]
    tl.store(DA_ptr, dA)


class SelectiveScan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, start, grad_detach, initial_state=None):
        """
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)
        start: (b, l, 1)
        """
        b_size, T, d = u.shape
        K = B.shape[-1]

        ctx.b_size = b_size
        ctx.T = T
        ctx.d = d
        ctx.K = K
        BV = 64
        num_warps = 4

        if b_size <= 16:
            BV = 32
            num_warps = 2

        NV = triton.cdiv(d, BV)

        o = torch.empty_like(u)
        H = torch.empty(b_size, T, d, K, device=u.device, dtype=torch.float32)

        if initial_state is None:
            initial_state = torch.zeros(
                b_size, d, K, device=u.device, dtype=torch.float32)
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        delta = delta.contiguous()
        u = u.contiguous()
        o = o.contiguous()
        H = H.contiguous()
        start = start.contiguous()
        initial_state = initial_state.contiguous()
        grad_detach = grad_detach.contiguous()
        fwd_recurrence[(b_size, NV)](A, B, C, delta, u, o, H, start,
                                     initial_state, T, d, K, BV, num_warps=num_warps, num_stages=1)
        o = reduce(H, C)
        ctx.save_for_backward(A, B, C, delta, H, u, start, grad_detach)
        ctx.initial_state = initial_state
        return o, H[:, -1]

    @staticmethod
    def backward(ctx, grad_output, d_final_state):
        do = grad_output
        A, B, C, delta, H, u, start, grad_detach = ctx.saved_tensors
        b_size = ctx.b_size
        T = ctx.T
        d = ctx.d
        K = ctx.K

        BV = 64
        num_warps = 4

        if b_size <= 16:
            BV = 32
            num_warps = 2

        NV = triton.cdiv(d, BV)
        dA = A.new_empty(b_size, d, K)
        du = torch.empty_like(u)
        d_delta = torch.empty_like(delta)
        db = B.new_empty(NV, b_size, T, K)
        dc = C.new_empty(NV, b_size, T, K)

        bwd_recurrence[(b_size, NV)](A, B, C, u, delta, do, H, start,
                                     dA, db, dc,
                                     d_delta, du, b_size, ctx.initial_state, grad_detach, T, d, K, BV, num_warps=num_warps)
        # dA = dA / valid_num
        db = db.sum(0)
        dc = dc.sum(0)

        return du, d_delta, dA.sum(0), db, dc, None, None, None


def triton_selective_scan_sequential(u, delta, A, B, C, D, start, grad_detach, initial_state=None):
    """
    u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    delta: shape (b, l, d_in)
    A: shape (d_in, n)
    B: shape (b, l, n)
    C: shape (b, l, n)
    D: shape (d_in,)
    start: (b, l, 1)
    """
    original_dtype = u.dtype
    D = D.float()
    A = A.float()
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = SelectiveScan.apply(u, delta, A, B, C, start, grad_detach, initial_state)
    o = o + D * u
    return o.to(original_dtype), final_state

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function
import math


@triton.jit
def _complex_operator_element_(_x_real_a, _a_imag_a,
                         _x_imag_a, _a_real_a, start, num,
                               interval, offset_b, offset_n, L, C, last_interval,
                         BLOCK_M: tl.constexpr):
    offset_t = tl.program_id(0)
    # Compute the thread index
    range_batch = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M
    range_time = (tl.arange(0, num) * interval + start) * C
    range_2dim = range_batch[:, None] + range_time[None, :]
    # range_2dim = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M + (offset_t * interval + start) * C
    ptr = range_2dim
    ptr_last = range_2dim - last_interval * C
    x_real_a = tl.load(_x_real_a + ptr).to(tl.float32)
    x_real_a_last = tl.load(_x_real_a + ptr_last).to(tl.float32)
    a_imag_a = tl.load(_a_imag_a + ptr).to(tl.float32)
    a_imag_a_last = tl.load(_a_imag_a + ptr_last).to(tl.float32)
    x_imag_a = tl.load(_x_imag_a + ptr).to(tl.float32)
    x_imag_a_last = tl.load(_x_imag_a + ptr_last).to(tl.float32)
    a_real_a = tl.load(_a_real_a + ptr).to(tl.float32)
    a_real_a_last = tl.load(_a_real_a + ptr_last).to(tl.float32)
    x_real_a = x_real_a + a_real_a * x_real_a_last - a_imag_a * x_imag_a_last
    x_imag_a = x_imag_a + a_real_a * x_imag_a_last + a_imag_a * x_real_a_last
    tl.store(_x_real_a + ptr, x_real_a.to(_x_real_a.dtype.element_ty))
    tl.store(_x_imag_a + ptr, x_imag_a.to(_x_imag_a.dtype.element_ty))

    a_real_a_next = a_real_a * a_real_a_last - a_imag_a * a_imag_a_last
    a_imag_a_next = a_imag_a * a_real_a_last - a_real_a * a_imag_a_last

    tl.store(_a_real_a + ptr, a_real_a_next.to(_a_real_a.dtype.element_ty))
    tl.store(_a_imag_a + ptr, a_imag_a_next.to(_a_imag_a.dtype.element_ty))



@triton.jit
def fwd_sequential_scan_complex(
    v_real,
    v_imag,
    decay_real,
    decay_imag,
    hidden_real,
    hidden_imag,
    hidden_real_input,
    hidden_imag_input,
    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):
    
    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M        
    ptr_input_hidden = tl.arange(0, BLOCK_M) + offset_b * C + offset_n * BLOCK_M
    # h_real = tl.zeros([BLOCK_M,], dtype=tl.float32)
    # h_imag = tl.zeros([BLOCK_M,], dtype=tl.float32)

    h_real = tl.load(hidden_real_input + ptr_input_hidden).to(tl.float32)
    h_imag = tl.load(hidden_imag_input + ptr_input_hidden).to(tl.float32)

    for _ in range(L):        
        x_real = tl.load(v_real + ptr).to(tl.float32)                
        x_imag = tl.load(v_imag + ptr).to(tl.float32)
        
        f_real = tl.load(decay_real + ptr).to(tl.float32) 
        f_imag = tl.load(decay_imag + ptr).to(tl.float32) 
        
        h_real_new = h_real * f_real - h_imag * f_imag + x_real
        h_imag_new = h_real * f_imag + h_imag * f_real + x_imag

        tl.store(hidden_real + ptr, h_real_new.to(hidden_real.dtype.element_ty))
        tl.store(hidden_imag + ptr, h_imag_new.to(hidden_imag.dtype.element_ty))
        h_real = h_real_new
        h_imag = h_imag_new
        ptr += C


@triton.jit
def bwd_sequential_scan_complex(

    grad_output_real,
    grad_output_imag,

    v_real,
    v_imag,

    f_real,
    f_imag,
        
    hidden_real,
    hidden_imag,

    grad_detach,

    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):

    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)    

    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + (L-1) * C + offset_n * BLOCK_M
    # ptr_1d = tl.arange(0, BLOCK_M) + offset_b * L + (L-1) * C + offset_n * BLOCK_M
    grad_detach_ptr = grad_detach + offset_b * L + (L - 1)
    grad_h_real = tl.zeros([BLOCK_M,], dtype=tl.float32)
    grad_h_imag = tl.zeros([BLOCK_M,], dtype=tl.float32)

    for time_step in range(L-1, -1, -1):  # L-1, L-2, ..., 0
        # basic: grad_h_real
        # grad_h_real[L] grad_h_imag[L] = 0
        # grad_h_real[t] = grad_real[t] + grad_h_real[t+1] * decay_real[t+1] + grad_h_image[t+1] * decay_imag[t+1]
        # grad_h_imag[t] = grad_imag[t] + grad_h_imag[t+1] * decay_real[t+1] - grad_h_real[t+1] * decay_imag[t+1]
        # advanced: grad_f_real
        # the one before h_real[0], h_imag[0] = 0
        # grad_f_real[t] = grad_h_real[t] * h_real[t-1] + grad_h_image[t] * h_imag[t-1]
        # grad_f_imag[t] = grad_h_real[t] * h_imag[t-1] - grad_h_image[t] * h_real[t-1]
        grad_real = tl.load(grad_output_real + ptr).to(tl.float32)
        grad_imag = tl.load(grad_output_imag + ptr).to(tl.float32)

        grad_detach_item = tl.load(grad_detach_ptr).to(tl.float32)

        grad_h_real = grad_h_real * (1 - grad_detach_item)
        grad_h_imag = grad_h_imag * (1 - grad_detach_item)

        grad_h_real += grad_real
        grad_h_imag += grad_imag
        
        decay_real = tl.load(f_real + ptr).to(tl.float32)   
        decay_imag = tl.load(f_imag + ptr).to(tl.float32)   
        # TODO: set other to the first hidden state
        # TODO: bug here!!
        # L-2, L-3, ..., 0, None
        h_real = tl.load(hidden_real + ptr).to(tl.float32)
        h_imag = tl.load(hidden_imag + ptr).to(tl.float32)

        grad_f_real = (grad_h_real * h_real + grad_h_imag * h_imag) 
        grad_f_imag = (grad_h_imag * h_real - grad_h_real * h_imag) 

        tl.store(f_real + ptr, grad_f_real.to(f_real.dtype.element_ty))                
        tl.store(f_imag + ptr, grad_f_imag.to(f_real.dtype.element_ty))                

        tl.store(v_real + ptr, grad_h_real.to(v_real.dtype.element_ty))
        tl.store(v_imag + ptr, grad_h_imag.to(v_real.dtype.element_ty))

        grad_h_real_new = grad_h_real * decay_real + grad_h_imag * decay_imag 
        grad_h_imag_new = grad_h_imag * decay_real - grad_h_real * decay_imag
        
        grad_h_real = grad_h_real_new
        grad_h_imag = grad_h_imag_new
        
        ptr -= C
        grad_detach_ptr -= 1



class TritonSequentialScan_Complex(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, v_real, v_imag, f_real, f_imag, hidden_real_input, hidden_imag_input, grad_detach):
        B,L,C = v_real.shape
        num_warps = 8
        assert C % 256 == 0, 'Hidden dimension must be multiple of 256'
        v_real = v_real.contiguous()
        v_imag = v_imag.contiguous()
        f_real = f_real.contiguous()
        f_imag = f_imag.contiguous()

        hidden_real_input = hidden_real_input.contiguous()
        hidden_imag_input = hidden_imag_input.contiguous()

        hidden_real = torch.zeros_like(v_real).contiguous()
        hidden_imag = torch.zeros_like(v_imag).contiguous()
        fwd_sequential_scan_complex[(B, int(C/256))](
            v_real,
            v_imag,
            f_real,
            f_imag,
            hidden_real,
            hidden_imag,
            hidden_real_input,
            hidden_imag_input,
            B,
            L,
            C, 
            BLOCK_M=256,
            num_warps=num_warps
        )

        ctx.save_for_backward(v_real, v_imag, f_real, f_imag, hidden_real, hidden_imag, hidden_real_input, hidden_imag_input, grad_detach)
        return hidden_real, hidden_imag
            
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output_real, grad_output_imag):
        
        v_real, v_imag, f_real, f_imag, hidden_real, hidden_imag, hidden_real_input, hidden_imag_input, grad_detach = ctx.saved_tensors
        B, L, C = v_real.shape
        
        num_warps = 8
        hidden_real = torch.cat((hidden_real_input[..., :1, :], hidden_real[..., :-1, :]), dim=-2)
        hidden_imag = torch.cat((hidden_imag_input[..., :1, :], hidden_imag[..., :-1, :]), dim=-2)

        bwd_sequential_scan_complex[(B,  int(C/256))](
            grad_output_real, 
            grad_output_imag,

            v_real, 
            v_imag,
            f_real,
            f_imag,


            hidden_real, 
            hidden_imag,

            grad_detach,

            B,
            L,
            C, 
            BLOCK_M=256,
            num_warps=num_warps
        )
        return v_real, v_imag, f_real, f_imag, None, None, None

complex_scan = TritonSequentialScan_Complex.apply




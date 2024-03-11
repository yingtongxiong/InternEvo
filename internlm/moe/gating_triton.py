from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl

# from generateTrace import _capacity

# from .gshard_moe import _capacity, gumbel_rsample
# @torch.jit.script
# def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
#     # gates has shape of SE
#     num_tokens = gates.shape[0]
#     num_experts = gates.shape[1]
#     # to(torch.int64) works around a bug in torch.onnx.export:
#     # it should cast k to int64 when converting torch.topk but it doesn't.
#     capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
#     # import pdb; pdb.set_trace()
#     if capacity < min_capacity:
#         capacity = min_capacity.to(torch.int64)
#     return capacity

# def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
#     gumbel = gumbel_map.get(device)
#     if gumbel is None:
#         one = torch.tensor(1.0, device=device)
#         zero = torch.tensor(0.0, device=device)
#         gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
#         gumbel_map[device] = gumbel
#     return gumbel(shape)

'''
This python file provide a triton implementation for top2gating in moe.
We fuse the top2gating into 3 kernels to reduce the kernel launch overhead in origin implementation.
'''

@triton.jit
def _fused_fwd_kernel1(
    logits_ptr, # input
    logits_w_noise_ptr, # input
    mask1_ptr, # output
    mask2_ptr, # output
    gates_ptr, # output
    fill_value,
    stride_logits_row,                                                                    
    stride_mask_row,
    expert_num: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr,
):
    
    '''
    Thie kernel fuses softmax + argmax + masked_fill + one_hot
    '''
    
    # the softmax computation for each row is independent
    # each block process each row
    pid = tl.program_id(axis=0)
    logits1_col = tl.arange(0, BLOCK_SIZE_e)
    logits2_col = tl.arange(0, BLOCK_SIZE_e)
    
    logits1_ptrs = logits_ptr + pid * stride_logits_row + logits1_col
    gates_ptrs = gates_ptr + pid * stride_logits_row + logits1_col
    
    # load data
    logits1_data = tl.load(logits1_ptrs, mask=logits1_col < expert_num, other=-float("inf"))
    logits1_exp = tl.exp(logits1_data)
    denom1 = tl.sum(logits1_exp, axis=0)
    softmax1_output = logits1_exp / denom1
    argmax1_output = tl.argmax(softmax1_output, axis=0)
    tl.store(gates_ptrs, softmax1_output)
    
    logits2_ptrs = logits_w_noise_ptr + pid * stride_logits_row + argmax1_output
    
    logits2_ptrs = logits_w_noise_ptr + pid * stride_logits_row + logits2_col
    logits2_data = tl.load(logits2_ptrs, mask=logits1_col < expert_num and logits1_col != argmax1_output, other=fill_value)
    argmax2_output = tl.argmax(logits2_data, axis=0)
    
    # compute the output pointer
    mask1_ptrs = mask1_ptr + pid * stride_mask_row + argmax1_output
    mask2_ptrs = mask2_ptr + pid * stride_mask_row + argmax2_output
    # store data
    tl.store(mask1_ptrs, 1, mask=argmax1_output < expert_num)
    tl.store(mask2_ptrs, 1, mask=argmax2_output < expert_num)

@triton.jit
def _fused_fwd_kernel2(
    mask1_ptr, # input
    mask2_ptr, # input
    gates_ptr, # input
    locations1, # output: locations1 * mask1
    locations2, # output: locations2 * mask2
    res, # output: me * ce
    stride_mask_row,
    stride_gates_row,
    stride_location_row,
    capacity: tl.constexpr,
    seq_len: tl.constexpr, expert_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    
    '''
    This kernel fuses sum + mean + lt + *
    '''
    
    pid = tl.program_id(axis=0)
    
    mask_row = tl.arange(0, BLOCK_SIZE)
    
    mask1_ptrs = mask1_ptr + mask_row * stride_mask_row + pid
    mask2_ptrs = mask2_ptr + mask_row * stride_mask_row + pid
    gates_ptrs = gates_ptr + mask_row * stride_gates_row + pid
    
    mask1_data = tl.load(mask1_ptrs, mask=mask_row < seq_len)
    mask2_data = tl.load(mask2_ptrs, mask=mask_row < seq_len)
    gates_data = tl.load(gates_ptrs, mask=mask_row < seq_len)
    
    mask1_sum = tl.sum(mask1_data, axis=0)
    loca1 = tl.cumsum(mask1_data, axis=0) - 1
    loca2 = tl.cumsum(mask2_data, axis=0) - 1 + mask1_sum
    
    loca1_ptrs = locations1 + mask_row * stride_location_row + pid
    loca2_ptrs = locations2 + mask_row * stride_location_row + pid
    
    me = tl.sum(gates_data, axis=0) / seq_len
    ce = tl.sum(mask1_data, axis=0) / seq_len 
    mul = me * ce * expert_num * expert_num
    
    res_ptrs = res + pid
    
    mask1_data *= tl.where(loca1 < capacity, 1, 0)
    mask2_data *= tl.where(loca2 < capacity, 1, 0)
    
    loca1 *= mask1_data
    loca2 *= mask2_data
    
    tl.store(loca1_ptrs, loca1, mask=mask_row < seq_len)
    tl.store(loca2_ptrs, loca2, mask=mask_row < seq_len)
    tl.store(res_ptrs, mul, mask=pid < expert_num)
    tl.store(mask1_ptrs, mask1_data, mask=mask_row < seq_len)
    tl.store(mask2_ptrs, mask2_data, mask=mask_row < seq_len)


@triton.jit
def _fused_fwd_kernel3(
    gates_ptr, # input
    input1_ptr, # locations1 * mask1
    input2_ptr, # locations2 * mask2
    mask1_ptr, # input
    mask2_ptr, # input
    combine_ptr, # output
    dispatch_mask_ptr, # output
    stride_gates_s,
    stride_input_s,
    stride_mask_s,
    stride_combine_s, stride_combine_e,
    e: tl.constexpr, c: tl.constexpr,
    min_value: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr,
):
    '''
    This kernel fuses sum + einsum + div + one_hot
    '''
    
    s_pid = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE_e)
    
    input1_ptrs = input1_ptr + s_pid * stride_input_s + e_offset
    input2_ptrs = input2_ptr + s_pid * stride_input_s + e_offset
    
    input1_data = tl.load(input1_ptrs, mask=e_offset < e)
    input2_data = tl.load(input2_ptrs, mask=e_offset < e)
    
    locations1_s = tl.sum(input1_data, axis=0)
    locations2_s = tl.sum(input2_data, axis=0)

    
    gates_ptrs = gates_ptr + s_pid * stride_gates_s + e_offset
    mask1_ptrs = mask1_ptr + s_pid * stride_mask_s + e_offset
    mask2_ptrs = mask2_ptr + s_pid * stride_mask_s + e_offset
    
    gates_data = tl.load(gates_ptrs, mask=e_offset < e)
    mask1_data = tl.load(mask1_ptrs, mask=e_offset < e).to(tl.float32)
    mask2_data = tl.load(mask2_ptrs, mask=e_offset < e).to(tl.float32)
    
    multi1 = gates_data * mask1_data
    multi2 = gates_data * mask2_data
    gates1_s = tl.sum(multi1, axis=0)
    gates2_s = tl.sum(multi2, axis=0)
    denom_s = gates1_s + gates2_s
    denom_s = tl.where(denom_s < min_value, min_value, denom_s)
    gates1_s /= denom_s
    gates2_s /= denom_s
    
    gates1 = gates1_s * mask1_data
    gates2 = gates2_s * mask2_data

    if locations1_s == locations2_s:
        data = gates1 + gates2
        combine_ptrs = combine_ptr + s_pid * stride_combine_s + e_offset * stride_combine_e + locations1_s
        mask = (e_offset < e) & (locations1_s < c)
        tl.store(combine_ptrs, data, mask=mask)
        
        dispatch_mask_ptrs = dispatch_mask_ptr + s_pid * stride_combine_s + e_offset * stride_combine_e + locations1_s
        dispatch_mask_data = tl.where(data > 0, 1, 0)
        tl.store(dispatch_mask_ptrs, dispatch_mask_data, mask=mask)
    else:
        combine1_ptrs = combine_ptr + s_pid * stride_combine_s + e_offset * stride_combine_e + locations1_s
        combine2_ptrs = combine_ptr + s_pid * stride_combine_s + e_offset * stride_combine_e + locations2_s
        mask1_ = (e_offset < e) & (locations1_s < c)
        mask2_ = (e_offset < e) & (locations2_s < c)
        tl.store(combine1_ptrs, gates1, mask=mask1_)
        tl.store(combine2_ptrs, gates2, mask=mask2_)
        
        dispatch_mask_ptrs1 = dispatch_mask_ptr + s_pid * stride_combine_s + e_offset * stride_combine_e + locations1_s
        dispatch_mask_ptrs2 = dispatch_mask_ptr + s_pid * stride_combine_s + e_offset * stride_combine_e + locations2_s
        tl.store(dispatch_mask_ptrs1, tl.where(gates1 > 0, 1, 0), mask=mask1_)
        tl.store(dispatch_mask_ptrs2, tl.where(gates2 > 0, 1, 0), mask=mask2_)

def _fused_top2_gating_fwd(logits, logits_w_noise, capacity):
    # prepare outputs for kernels
    s, e = logits.shape
    # output for kernel1
    # mask1 = torch.zeros_like(logits).to(torch.int64)
    # mask2 = torch.zeros_like(logits).to(torch.int64)
    mask1 = torch.zeros((s, e), device=logits.device, dtype=torch.int64)
    mask2 = torch.zeros((s, e), device=logits.device, dtype=torch.int64)
    gates = torch.zeros_like(logits)
    # output for kernel2
    locations1 = torch.zeros_like(mask1)
    locations2 = torch.zeros_like(mask1)
    res = torch.zeros((e, ), device=logits.device)
    # output for kernel3
    combine_weights = torch.zeros((s, e, capacity), device=logits.device)
    dispatch_mask = torch.zeros((s, e, capacity), device=logits.device, dtype=torch.bool)
    
    # launch kernel1
    stride_logits_row, _ = logits.stride()
    stride_mask_row, _ = mask1.stride()
    block_size_e = triton.next_power_of_2(e)
    fill_value = torch.finfo(logits.dtype).min
    
    _fused_fwd_kernel1[(s, )](
        logits,
        logits_w_noise,
        mask1,
        mask2,
        gates,
        fill_value,
        stride_logits_row,
        stride_mask_row,
        e, block_size_e,
    )
    
    # launch kernel2
    block_size_s = triton.next_power_of_2(s)
    stride_gates_row, _ = gates.stride()
    stride_location_row, _ = locations1.stride()
    
    _fused_fwd_kernel2[(e, )](
        mask1,
        mask2,
        gates,
        locations1,
        locations2,
        res,
        stride_mask_row,
        stride_gates_row,
        stride_location_row,
        capacity,
        s, e, block_size_s,
    )
    
    # launch kernel3
    stride_combine_s, stride_combine_e, _ = combine_weights.stride()
    min_value = torch.finfo(gates.dtype).eps
    
    _fused_fwd_kernel3[(s, )](
        gates,
        locations1,
        locations2,
        mask1,
        mask2,
        combine_weights,
        dispatch_mask,
        stride_gates_row,
        stride_location_row,
        stride_mask_row,
        stride_combine_s,
        stride_combine_e,
        e, capacity,
        min_value,
        block_size_e,
    )
    
    return res, combine_weights, dispatch_mask

def fused_top2_gating_fwd(logits: Tensor, capacity_factor: float = 1.0, min_capacity: int = 2) -> Tuple[Tensor, Tensor, Tensor]:

    # capacity = _capacity(logits, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))
    logits_w_noise = logits #+ gumbel_rsample(logits.shape, device=logits.device)
    res, combine_weights, dispatch_mask = _fused_top2_gating_fwd(logits, logits_w_noise, capacity=2048)
    l_aux = torch.mean(res)
    
    return l_aux, combine_weights, dispatch_mask

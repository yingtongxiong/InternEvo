import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel1(
    logits1,
    logits2,
    mask1, # output
    mask2,
    gates,
    fill_value,
    stride_se_s,
    e: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr,
):
    # the softmax computation for each row is independent
    # each block process each row
    pid = tl.program_id(axis=0)
    logits1_col = tl.arange(0, BLOCK_SIZE_e)
    logits2_col = tl.arange(0, BLOCK_SIZE_e)
    
    logits1_ptrs = logits1 + pid * stride_se_s + logits1_col
    gates_ptrs = gates + pid * stride_se_s + logits1_col
    
    # load data
    logits1_data = tl.load(logits1_ptrs, mask=logits1_col < e, other=-float("inf"))
    logits1_exp = tl.exp(logits1_data)
    denom1 = tl.sum(logits1_exp, axis=0)
    softmax1_output = logits1_exp / denom1
    argmax1_output = tl.argmax(softmax1_output, axis=0)
    tl.store(gates_ptrs, softmax1_output, mask=logits1_col < e)
    
    logits2_ptrs = logits2 + pid * stride_se_s + argmax1_output
    logits2_ptrs = logits2 + pid * stride_se_s + logits2_col
    logits2_data = tl.load(logits2_ptrs, mask=logits1_col < e and logits1_col != argmax1_output, other=fill_value)
    argmax2_output = tl.argmax(logits2_data, axis=0)
    
    # compute the output pointer
    mask1_ptrs = mask1 + pid * stride_se_s + argmax1_output
    mask2_ptrs = mask2 + pid * stride_se_s + argmax2_output
    # store data
    tl.store(mask1_ptrs, 1, mask=argmax1_output < e)
    tl.store(mask2_ptrs, 1, mask=argmax2_output < e)

def fused_gating1(logits1, logits2):
    
    mask1 = torch.zeros_like(logits1).to(torch.int64)
    mask2 = torch.zeros_like(logits2).to(torch.int64)
    gates = torch.zeros_like(logits1)
    
    stride_se_s, _ = logits1.stride()
    s, e = logits1.shape
    
    block_size_e = triton.next_power_of_2(e)
    fill_value = torch.finfo(logits1.dtype).min
    
    _fused_kernel1[(s, )](
        logits1,
        logits2,
        mask1,
        mask2,
        gates,
        fill_value,
        stride_se_s,
        e,
        BLOCK_SIZE_e=block_size_e,
    )

    return mask1, mask2, gates

@triton.jit
def _fused_kernel2(
    gates, # input
    mask1,
    mask2,
    locations1,
    locations2,
    res,
    ce,
    stride_se_s,
    capacity: tl.constexpr,
    s: tl.constexpr, e: tl.constexpr,
    BLOCK_SIZE_s: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    mask_row = tl.arange(0, BLOCK_SIZE_s)
    
    mask1_ptrs = mask1 + mask_row * stride_se_s + pid
    mask2_ptrs = mask2 + mask_row * stride_se_s + pid
    gates_ptrs = gates + mask_row * stride_se_s + pid
    
    mask1_data = tl.load(mask1_ptrs, mask=mask_row < s)
    mask2_data = tl.load(mask2_ptrs, mask=mask_row < s)
    gates_data = tl.load(gates_ptrs, mask=mask_row < s)
    
    mask1_sum = tl.sum(mask1_data, axis=0)
    loca1 = tl.cumsum(mask1_data, axis=0) - 1
    loca2 = tl.cumsum(mask2_data, axis=0) - 1 + mask1_sum
    
    loca1_ptrs = locations1 + mask_row * stride_se_s + pid
    loca2_ptrs = locations2 + mask_row * stride_se_s + pid
    
    me = tl.sum(gates_data, axis=0) / s
    ce_data = tl.sum(mask1_data, axis=0) / s 
    mul = me * ce_data * e * e
    
    res_ptrs = res + pid
    ce_ptrs = ce + pid
    
    mask1_data *= tl.where(loca1 < capacity, 1, 0)
    mask2_data *= tl.where(loca2 < capacity, 1, 0)
    
    loca1 *= mask1_data
    loca2 *= mask2_data
    
    tl.store(loca1_ptrs, loca1, mask=mask_row < s)
    tl.store(loca2_ptrs, loca2, mask=mask_row < s)
    tl.store(res_ptrs, mul, mask=pid < e)
    tl.store(ce_ptrs, ce_data, mask=pid < e)
    tl.store(mask1_ptrs, mask1_data, mask=mask_row < s)
    tl.store(mask2_ptrs, mask2_data, mask=mask_row < s)

def fused_gating2(mask1, mask2, gates, capacity=1):
    loca1 = torch.zeros_like(mask1)
    loca2 = torch.zeros_like(mask2)
    
    stride_se_s, _ = mask1.stride()
    
    s, e = mask1.shape
    block_size_s = triton.next_power_of_2(s)
    res = torch.zeros((e,)).to(mask1.device)
    
    _fused_kernel2[(e, )](
        gates,
        mask1,
        mask2,
        loca1,
        loca2,
        res,
        stride_se_s,
        capacity=capacity,
        s=s, e=e,
        BLOCK_SIZE_s=block_size_s,
    )
    
    return loca1, loca2, res, mask1, mask2

@triton.jit
def _fused_kernel3(
    gates,
    input1, # locations1 * mask1
    input2, # locations2 * mask2
    mask1,
    mask2,
    combine_weights,
    dispatch_mask,
    stride_se_s,
    stride_sec_s, stride_sec_e,
    e: tl.constexpr, c: tl.constexpr,
    min_value: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr,
):
    s_pid = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE_e)

    input1_ptrs = input1 + s_pid * stride_se_s + e_offset
    input2_ptrs = input2 + s_pid * stride_se_s + e_offset
    
    input1_data = tl.load(input1_ptrs, mask=e_offset < e)
    input2_data = tl.load(input2_ptrs, mask=e_offset < e)
    
    locations1_s = tl.sum(input1_data, axis=0)
    locations2_s = tl.sum(input2_data, axis=0)

    
    gates_ptrs = gates + s_pid * stride_se_s + e_offset
    mask1_ptrs = mask1 + s_pid * stride_se_s + e_offset
    mask2_ptrs = mask2 + s_pid * stride_se_s + e_offset
    
    gates_data = tl.load(gates_ptrs, mask=e_offset < e)
    mask1_data = tl.load(mask1_ptrs, mask=e_offset < e)
    mask2_data = tl.load(mask2_ptrs, mask=e_offset < e)
    
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
        combine_ptrs = combine_weights + s_pid * stride_sec_s + e_offset * stride_sec_e + locations1_s
        mask = (e_offset < e) & (locations1_s < c)
        tl.store(combine_ptrs, data, mask=mask)
        
        dispatch_mask_ptrs = dispatch_mask + s_pid * stride_sec_s + e_offset * stride_sec_e + locations1_s
        dispatch_mask_data = tl.where(data > 0, 1, 0)
        tl.store(dispatch_mask_ptrs, dispatch_mask_data, mask=mask)
    else:
        combine1_ptrs = combine_weights + s_pid * stride_sec_s + e_offset * stride_sec_e + locations1_s
        combine2_ptrs = combine_weights + s_pid * stride_sec_s + e_offset * stride_sec_e + locations2_s
        mask1_ = (e_offset < e) & (locations1_s < c)
        mask2_ = (e_offset < e) & (locations2_s < c)
        tl.store(combine1_ptrs, gates1, mask=mask1_)
        tl.store(combine2_ptrs, gates2, mask=mask2_)
        
        dispatch_mask_ptrs1 = dispatch_mask + s_pid * stride_sec_s + e_offset * stride_sec_e + locations1_s
        dispatch_mask_ptrs2 = dispatch_mask + s_pid * stride_sec_s + e_offset * stride_sec_e + locations2_s
        tl.store(dispatch_mask_ptrs1, tl.where(gates1 > 0, 1, 0), mask=mask1_)
        tl.store(dispatch_mask_ptrs2, tl.where(gates2 > 0, 1, 0), mask=mask2_)

def fused_gating3(gates, input1, input2, mask1, mask2, c=1):
    s, e = gates.shape
    stride_se_s, _ = gates.stride()
    
    combine_weights = torch.zeros((s, e, c), device=gates.device)
    dispatch_mask = torch.zeros((s, e, c), device=gates.device, dtype=torch.bool)
    
    stride_sec_s, stride_sec_e, _ = combine_weights.stride()
    
    min_value = torch.finfo(gates.dtype).eps
    BLOCK_SIZE_e = triton.next_power_of_2(e)
    
    _fused_kernel3[(s, )](
        gates,
        input1,
        input2,
        mask1,
        mask2,
        combine_weights,
        dispatch_mask,
        stride_se_s,
        stride_sec_s,
        stride_sec_e,
        e, c,
        min_value,
        BLOCK_SIZE_e,
    )
    
    return combine_weights, dispatch_mask


def _fused_top2gating(logits, logits_w_noise, capacity):
    mask1 = torch.zeros_like(logits).to(torch.int64)
    mask2 = torch.zeros_like(mask1)
    gates = torch.zeros_like(logits)
    
    stride_se_s, _ = logits.stride()
    s, e = logits.shape
    
    block_size_s = triton.next_power_of_2(s)
    block_size_e = triton.next_power_of_2(e)
    fill_value = torch.finfo(logits.dtype).min
    
    _fused_kernel1[(s, )](
        logits,
        logits_w_noise,
        mask1,
        mask2,
        gates,
        fill_value,
        stride_se_s,
        e,
        BLOCK_SIZE_e=block_size_e,
    )
    
    loca1 = torch.zeros_like(mask1)
    loca2 = torch.zeros_like(mask2)
    res = torch.zeros((e,)).to(mask1.device)
    ce = torch.zeros((e,)).to(mask1.device)
    
    _fused_kernel2[(e, )](
        gates,
        mask1,
        mask2,
        loca1,
        loca2,
        res,
        ce,
        stride_se_s,
        capacity=capacity,
        s=s, e=e,
        BLOCK_SIZE_s=block_size_s,
    )
    
    combine_weights = torch.zeros((s, e, capacity), device=gates.device)
    dispatch_mask = torch.zeros((s, e, capacity), device=gates.device, dtype=torch.bool)
    
    stride_sec_s, stride_sec_e, _ = combine_weights.stride()
    
    min_value = torch.finfo(gates.dtype).eps
    
    _fused_kernel3[(s, )](
        gates,
        loca1,
        loca2,
        mask1,
        mask2,
        combine_weights,
        dispatch_mask,
        stride_se_s,
        stride_sec_s,
        stride_sec_e,
        e, capacity,
        min_value,
        block_size_e,
    )
    
    return res, combine_weights, dispatch_mask, (loca1, loca2, mask1, mask2, gates, ce)


import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def _fused_bwd_kernel(
    grad_output1_ptr,
    locations1_sc_ptr,
    locations2_sc_ptr,
    ce_ptr,
    mask1_float_ptr,
    mask2_float_ptr,
    gates_ptr,
    diag_mask_ptr,
    grad_gates1_ptr,
    grad_gates2_ptr,
    grad_gates3_ptr,
    grad_gates_ptr,
    grad_denom_s_ptr,
    grad_logits_ptr,
    grad_output1_stride_s, grad_output1_stride_e,
    locations1_sc_stride_s,
    mask_stride_s,
    gates_stride_s,
    grad_output2: tl.constexpr,
    min_value: tl.constexpr,
    s: tl.constexpr, e: tl.constexpr, c: tl.constexpr,
    BLOCK_SIZE_c: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE_e)
    c_offset = tl.arange(0, BLOCK_SIZE_c)
    
    grad_output1_ptrs = grad_output1_ptr + pid * grad_output1_stride_s + e_offset[:, None] * grad_output1_stride_e + c_offset[None, :]
    locations1_sc_ptrs = locations1_sc_ptr + pid * locations1_sc_stride_s + c_offset
    locations2_sc_ptrs = locations2_sc_ptr + pid * locations1_sc_stride_s + c_offset
    mask1_float_ptrs = mask1_float_ptr + pid * mask_stride_s + e_offset
    mask2_float_ptrs = mask2_float_ptr + pid * mask_stride_s + e_offset
    gates_ptrs = gates_ptr + pid * gates_stride_s + e_offset
    ce_ptrs = ce_ptr + e_offset
    diag_mask_ptrs = diag_mask_ptr + e_offset[:, None] * e + e_offset[None, :]
    
    grad_mask = c_offset[None, :] < c and e_offset[:, None] < e
    grad_output1 = tl.load(grad_output1_ptrs, mask=grad_mask)
    locations1_sc = tl.load(locations1_sc_ptrs, mask=c_offset < c)
    locations2_sc = tl.load(locations2_sc_ptrs, mask=c_offset < c)
    mask1_float = tl.load(mask1_float_ptrs, mask=e_offset < e)
    mask2_float = tl.load(mask2_float_ptrs, mask=e_offset < e)
    gates = tl.load(gates_ptrs, mask=e_offset < e)
    ce = tl.load(ce_ptrs, mask=e_offset < e)
    diag_mask = tl.load(diag_mask_ptrs, mask=e_offset[:,None] < e and e_offset[None,:] < e)
    
    grad_gates1 = tl.sum(grad_output1 * locations1_sc, axis=1)
    grad_gates2 = tl.sum(grad_output1 * locations2_sc, axis=1)
    
    tl.device_print((locations1_sc).shape)
    
    grad_gates1_s = tl.sum(grad_gates1 * mask1_float, axis=0)
    grad_gates2_s = tl.sum(grad_gates2 * mask2_float, axis=0)
    
    # compute the gates1_s and gates2_s to re-compute the denom_s in forward
    gates1_s = tl.sum(gates * mask1_float, axis=0)
    gates2_s = tl.sum(gates * mask2_float, axis=0)
    
    denom_s = gates1_s + gates2_s
    denom_s_output = tl.where(denom_s < min_value, min_value, denom_s)
    grad_denom_s = -(grad_gates1_s * gates1_s + grad_gates2_s * gates2_s) / (denom_s * denom_s)
    grad_denom_s = tl.where(denom_s < min_value, 0, grad_denom_s)
    
    grad_gates1_s_ = grad_gates1_s / denom_s_output
    grad_gates2_s_ = grad_gates2_s / denom_s_output
    
    # grad_gates1_s_final = grad_gates1_s_ + grad_denom_s
    # grad_gates2_s_final = grad_gates2_s_ + grad_denom_s
    
    grad_gates1 = (grad_gates1_s_ + grad_denom_s) * mask1_float
    grad_gates2 = (grad_gates2_s_ + grad_denom_s) * mask2_float
    
    grad_me = grad_output2 * ce * e * e / e
    grad_gates_3 = (tl.zeros((e, ), dtype=tl.float32) + 1) * grad_me / s
    
    grad_gates = grad_gates1 + grad_gates2 + grad_gates_3
    
    # diag = tl.zeros((e, e), dtype=tl.float32)
    
    grad_gates_expand = tl.expand_dims(grad_gates, axis=0)
    gates_expand = tl.expand_dims(gates, axis=0)
    gates_in1 = tl.broadcast_to(gates_expand, (e, e))
    # tl.device_print(tl.trans(grad_gates_expand).shape)
    gates_in2 = tl.broadcast_to(tl.trans(gates_expand), (e, e))
    ger = gates_in1 * gates_in2
    softmax_grad = diag_mask * gates_in1 - ger
    grad_logits = tl.sum(softmax_grad * tl.broadcast_to(grad_gates_expand, (e, e)), axis=1)
    # tl.device_print(grad_logits.shape)
    
    grad_gates1_ptrs = grad_gates1_ptr + pid * gates_stride_s + e_offset
    grad_gates2_ptrs = grad_gates2_ptr + pid * gates_stride_s + e_offset
    grad_denom_s_ptrs = grad_denom_s_ptr + pid
    grad_gates3_ptrs = grad_gates3_ptr + pid * gates_stride_s + e_offset
    grad_gates_ptrs = grad_gates_ptr + pid * gates_stride_s + e_offset
    grad_logits_ptrs = grad_logits_ptr + pid * gates_stride_s + e_offset
    
    tl.store(grad_gates1_ptrs, grad_gates1, mask=pid < s)
    tl.store(grad_gates2_ptrs, grad_gates2, mask=pid < s)
    tl.store(grad_denom_s_ptrs, grad_denom_s, mask=pid < s)
    tl.store(grad_gates3_ptrs, grad_gates_3, mask=e_offset < e)
    tl.store(grad_gates_ptrs, grad_gates, mask=e_offset < e)
    tl.store(grad_logits_ptrs, grad_logits)

def fused_bwd(grad_output1, grad_output2, locations1_sc, locations2_sc, mask1_float, mask2_float, gates, ce):
    s, e, c = grad_output1.shape
    block_size_c = triton.next_power_of_2(c)
    block_size_e = triton.next_power_of_2(e)
    
    grad_gates1 = torch.zeros((s,e), device=grad_output1.device)
    grad_gates2 = torch.zeros((s,e), device=grad_output1.device)
    grad_gates3 = torch.zeros((s,e), device=grad_output1.device)
    grad_gates = torch.zeros((s,e), device=grad_output1.device)
    grad_logits = torch.zeros((s,e), device=grad_output1.device)
    grad_denom_s = torch.zeros((s,), device=grad_output1.device)
    diag_mask = torch.diag(torch.ones(e)).to(device=grad_output1.device)
    
    
    grad_output_stride_s, grad_output_stride_e, _ = grad_output1.stride()
    locations_stride_s, _ = locations1_sc.stride()
    # grad_gates_stride_s, _ = grad_gates1.stride()
    mask_stride_s, _ = mask1_float.stride()
    gates_stride_s, _ = gates.stride()
    
    _fused_bwd_kernel[(s, )](
        grad_output1,
        locations1_sc, 
        locations2_sc,
        ce,
        mask1_float,
        mask2_float,
        gates,
        diag_mask,
        grad_gates1,
        grad_gates2,
        grad_gates3,
        grad_gates,
        grad_denom_s,
        grad_logits,
        grad_output_stride_s, grad_output_stride_e,
        locations_stride_s,
        mask_stride_s,
        gates_stride_s,
        grad_output2,
        torch.finfo(gates.dtype).eps,
        s, e, c,
        block_size_c, block_size_e
    )
    
    return grad_gates, grad_logits, grad_gates3

def baseline(grad_output1, grad_output2, locations1_sc, locations2_sc, mask1_float, mask2_float, gates, ce):
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)
    gates2_s = torch.einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s_output = torch.clamp(denom_s, min=torch.finfo(gates.dtype).eps)
    grad_gates1 = torch.einsum("sec,sc->se", grad_output1, locations1_sc)
    grad_gates2 = torch.einsum("sec,sc->se", grad_output1, locations2_sc)
    
    grad_gates1_s = torch.einsum("se,se->s", grad_gates1, mask1_float)
    grad_gates2_s = torch.einsum("se,se->s", grad_gates2, mask2_float)
    
    grad_denom_s = -(grad_gates1_s * gates1_s + grad_gates2_s * gates2_s) / (denom_s ** 2)
    grad_denom_s[denom_s < torch.finfo(gates.dtype).eps] = 0
    
    grad_gates1_s_ = grad_gates1_s / denom_s_output
    grad_gates2_s_ = grad_gates2_s / denom_s_output
    
    grad_gates1_s_final = grad_gates1_s_ + grad_denom_s
    grad_gates2_s_final = grad_gates2_s_ + grad_denom_s
    
    grad_gates1 = torch.einsum("s,se->se", grad_gates1_s_final, mask1_float)
    grad_gates2 = torch.einsum("s,se->se", grad_gates2_s_final, mask2_float)
    
    s, e, c = grad_output1.shape
    grad_me = grad_output2 * e * e * ce /  ce.shape[0]
    grad_gates3 = grad_me / s * torch.ones((s, e), device=ce.device)
    
    grad_gates = grad_gates1 + grad_gates2 + grad_gates3
    
    grad_logits = []
    for i in range(grad_gates.shape[0]):
        softmax_grad = torch.diag(gates[i]) - torch.ger(gates[i], gates[i])
        grad_logits.append(torch.matmul(softmax_grad, grad_gates[i].t()))
    grad_logits = torch.stack(grad_logits)
    
    return grad_gates, grad_logits, grad_gates3

def test():
    device = torch.device("cuda:0")
    s, e, c = 4 * 1024, 4, 2 * 1024
    grad_output = torch.randn((s, e, c), device=device)
    locations1_sc = torch.randn((s, c), device=device)
    locations2_sc = torch.randn((s, c), device=device)
    # mask1 = torch.randn((s, e), device=device)
    # mask2 = torch.randn((s, e), device=device)
    logits = torch.randn((s, e), device=device)
    gates = F.softmax(logits, dim=1)
    indices1 = torch.argmax(gates, dim=1)
    mask1 = F.one_hot(indices1, num_classes=e)
    logits2 = logits.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
    indices2 = torch.argmax(logits2, dim=1)
    mask2 = F.one_hot(indices2, num_classes=e)
    
    
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # locations2 += torch.sum(mask1, dim=0, keepdim=True)
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    # import pdb; pdb.set_trace()
    locations1_sc = F.one_hot(locations1_s, num_classes=c).type_as(logits)
    locations2_sc = F.one_hot(locations2_s, num_classes=c).type_as(logits)
    
    mask1 = mask1.to(torch.float32)
    mask2 = mask2.to(torch.float32)
    
    grad_output2 = 1.3
    ce = torch.mean(mask1, dim=0)
    
    
    output1, output2, output3 = baseline(grad_output, grad_output2, locations1_sc, locations2_sc, mask1, mask2, gates, ce)
    output1_triton, output2_triton, output3_triton = fused_bwd(grad_output, grad_output2, locations1_sc, locations2_sc, mask1, mask2, gates, ce)
    
    atol = 1e-6
    assert output1.shape == output1_triton.shape
    
    # import pdb; pdb.set_trace()
    assert torch.allclose(output1, output1_triton, atol=atol)
    assert output2.shape == output2_triton.shape
    assert torch.allclose(output2, output2_triton, atol=atol)
    assert output3.shape == output3_triton.shape
    # import pdb; pdb.set_trace()
    assert torch.allclose(output3, output3_triton, atol=atol)

test()
    
    
    
    
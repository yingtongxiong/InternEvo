from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor

from testTriton import fused_gating1, fused_gating2, fused_gating3, fused_gating5

def softmax_argmax(input: torch.Tensor):
    softmax_output = F.softmax(input, dim=1)
    argmax_output = torch.argmax(softmax_output, dim=1)
    return argmax_output

gumbel_map: Dict[torch.device, Callable] = {}

@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity

def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)

def top2gating(logits: Tensor, capacity_factor: float = 1.0, min_capacity: int = 2) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    capacity = _capacity(logits, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))
    # logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    gates = F.softmax(logits, dim=1)
    
    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    
    # Replace top-expert with min value
    # logits_except1 = logits_w_noise.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
    logits_except1 = logits.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)
    
    # return mask1, mask2, gates

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.type_as(logits), dim=0)
    res = me * ce * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    
    l_aux = torch.mean(res)

    # Store the capacity location for each token
    tmp1 = locations1 * mask1
    tmp2 = locations2 * mask2
    locations1_s = torch.sum(tmp1, dim=1)
    locations2_s = torch.sum(tmp2, dim=1)
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity).type_as(logits)
    
    # return l_aux, locations1_sc, locations2_sc

    # Normalize gate probabilities
    mask1_float = mask1.type_as(logits)
    mask2_float = mask2.type_as(logits)
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)
    gates2_s = torch.einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
    gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
    
    combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask

def top2gating_triton(logits: Tensor, capacity_factor: float = 1.0, min_capacity: int = 2) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    capacity = _capacity(logits, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity)).item()
    # logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    num_experts = int(logits.shape[1])
    
    mask1, mask2, gates = fused_gating1(logits, logits)
    
    # return mask1, mask2, gates
    
    locations1, locations2, res, mask1, mask2 = fused_gating2(mask1, mask2, gates, capacity)
    l_aux = torch.mean(res)
    # Store the capacity location for each token
    locations1_sc, locations2_sc = fused_gating3(locations1, locations2, capacity=capacity)
    
    # return l_aux, locations1_sc, locations2_sc
    
    combine_weights, dispatch_mask = fused_gating5(gates, mask1, mask2, locations1_sc, locations2_sc)

    return l_aux, combine_weights, dispatch_mask

iteration = 10

device = torch.device('cuda:0')
shape = (4096, 4)
input = torch.randn(shape, device=device)

l_aux_torch, combine_weights_torch, dispatch_mask_torch = top2gating(input)
l_aux_triton, combine_weights_triton, dispatch_mask_triton = top2gating_triton(input)

assert l_aux_torch.shape == l_aux_triton.shape
assert torch.allclose(l_aux_torch, l_aux_triton)
assert combine_weights_torch.shape == combine_weights_triton.shape
assert torch.allclose(combine_weights_torch, combine_weights_triton)
assert dispatch_mask_torch.shape == dispatch_mask_triton.shape
assert torch.allclose(dispatch_mask_torch, dispatch_mask_triton)



with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("test",
        ),
        with_stack=True,
        with_modules=True,
        profile_memory=True,
    ) as prof:
    
        
        for i in range(0, iteration):
            # output = softmax_argmax(input)
            # output = fused_dim1(input)
            # output = top2gating(input)
            output = top2gating_triton(input)
            
            if i % 2 == 0:
                prof.step()

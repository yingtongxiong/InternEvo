from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from gating_triton import Top2GatingFunc, _capacity

def top2gating(logits: Tensor, noise: Tensor, capacity_factor: float = 1.0, min_capacity: int = 2) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    logits_w_noise = logits + noise
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
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


class Top2Gating(torch.nn.Module):
    
    def forward(self, logits, noise, capacity_factor: float = 1.0, min_capacity: int = 2):
        return top2gating(logits, noise, capacity_factor, min_capacity)

def test():
    device = torch.device('cuda:0')
    shape = (4096 * 2, 16)
    logits_torch = torch.randn(shape, device=device, requires_grad=True)
    noise = torch.randn(shape, device=device)
    
    with torch.no_grad():
        logits_triton = logits_torch.clone()

    logits_triton.requires_grad = True
    
    model_torch = Top2Gating()
    model_triton = Top2GatingFunc.apply
    
    output1_torch, output2_torch, output3_torch = model_torch(logits_torch, noise)
    output1_triton, output2_triton, output3_triton, _ = model_triton(logits_triton, noise)
    
    assert output1_torch.shape == output1_triton.shape
    # import pdb; pdb.set_trace()
    assert torch.allclose(output1_torch, output1_triton)
    assert output2_torch.shape == output2_triton.shape
    assert torch.allclose(output2_torch, output2_triton)
    assert output3_torch.shape == output3_triton.shape
    assert torch.allclose(output3_torch, output3_triton)
    
    # loss_torch = torch.mean(output1_torch * torch.sum(output2_torch + output3_torch, dim=0))
    # loss_triton = torch.mean(output1_triton * torch.sum(output2_triton + output3_triton, dim=0))
    
    loss_torch = torch.sum(torch.mean(output1_torch * output2_torch))
    loss_triton = torch.sum(torch.mean(output1_triton * output2_triton))
    
    assert torch.allclose(loss_torch, loss_triton)
    
    loss_torch.backward()
    loss_triton.backward()
    
    assert logits_torch.grad.shape == logits_triton.grad.shape
    assert torch.allclose(logits_torch.grad, logits_triton.grad)
    
    # l_aux_torch, combine_weights_torch, dispatch_mask_torch = top2gating(logits)
    # l_aux_triton, combine_weights_triton, dispatch_mask_triton = Top2GatingFunc.apply(logits)
    
    # assert l_aux_torch.shape == l_aux_triton.shape
    # assert torch.allclose(l_aux_torch, l_aux_triton)
    # assert combine_weights_torch.shape == combine_weights_triton.shape
    # assert torch.allclose(combine_weights_torch, combine_weights_triton)
    # assert dispatch_mask_torch.shape == dispatch_mask_triton.shape
    # assert torch.allclose(dispatch_mask_torch, dispatch_mask_triton)

test()
print("sucessfully!")
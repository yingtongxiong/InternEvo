from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from .gating_fwd_triton import _fused_top2gating
from .gating_bwd_triton import fused_bwd

# from gshard_moe import _capacity, gumbel_rsample
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

class Top2GatingFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: torch.Any, logits: torch.Tensor, noise: torch.Tensor, capacity_factor: float = 1.0, min_capacity: int = 2):
        # compute the capacity
        capacity = _capacity(logits, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity)).item()
        # add noise to the original logits
        # logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
        
        res, combine_weights, dispatch_mask, saved_tensors = _fused_top2gating(logits, noise, capacity)
        l_aux = torch.mean(res)
        ctx.save_for_backward(*saved_tensors)
        return l_aux, combine_weights, dispatch_mask, None
    
    @staticmethod
    def backward(ctx: torch.Any, *grad_outputs: torch.Any) -> torch.Any:
        grad_l_aux = grad_outputs[0].item()
        grad_combine = grad_outputs[1]
        
        loca1, loca2, mask1, mask2, gates, ce = ctx.saved_tensors
        
        grad_logits = fused_bwd(grad_l_aux, grad_combine, loca1, loca2, mask1, mask2, gates, ce)
        
        return grad_logits, None, None, None
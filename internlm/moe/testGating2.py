from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

# from internlm.core.context import ParallelMode
# from internlm.core.context import global_context as gpc
# from internlm.model.linear import FeedForward
# from internlm.utils.common import get_current_device
# from internlm.utils.logger import get_logger
# from internlm.utils.megatron_timers import megatron_timer as timer
# from internlm.utils.registry import MODEL_INITIALIZER


def top2gating(logits: Tensor, capacity_factor: float, min_capacity: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to("cpu")

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.type_as(logits), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.type_as(logits)
    mask2_float = mask2.type_as(logits)
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity).type_as(logits)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts

from torch._dynamo import optimize
import logging

def fn(x, y):
    a = torch.sin(x).cuda()
    b = torch.sin(y).cuda()
    return a + b

def einsum(rule: str, a: Tensor, b: Tensor, flag: bool = True):
    if flag:
        return torch.einsum(rule, a, b)
    elif rule == "s,se->se":
        # [1, s] * [s, e]
        return a.reshape(a.shape[0], -1) * b
    elif rule == "se,sc->sec":
        # [s,e,1] * [s,1,c]
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == "se,se->s":
        # [s,1,e] * [s,e,1]
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == "sec,sm->ecm":
        # [e*c, s] * [s, m]
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == "sec,ecm->sm":
        # [s, e*c] * [e*c, m]
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == "ks,ksm->sm":
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)

def fn2(mask1, mask2, gates, logits):
    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.type_as(logits), dim=0)
    l_aux = torch.mean(me * ce) * 4 * 4

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, 1024)
    mask2 *= torch.lt(locations2, 1024)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.to(logits.dtype) # mask1.type_as(logits)
    mask2_float = mask2.to(logits.dtype) # mask2.type_as(logits)
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    # locations1_sc = F.one_hot(locations1_s, num_classes=512).type_as(logits)
    # locations2_sc = F.one_hot(locations2_s, num_classes=512).type_as(logits)
    # combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
    # combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
    # combine_weights = combine1_sec + combine2_sec
    # dispatch_mask = combine_weights.bool()

torch._logging.set_logs(dynamo = logging.DEBUG)
torch._inductor.config.trace.enabled = True
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

new_fn = torch.compile(fn2, backend="inductor")
# new_fn = optimize("inductor")(fn)
input_tensor1 = torch.randn(512, 512).to(device="cuda:0")#.to(dtype=torch.bool)
input_tensor2 = torch.randn(512, 512).to(device="cuda:0")#.to(dtype=torch.bool)
input_tensor3 = torch.randn(512, 512).to(device="cuda:0")
input_tensor4 = torch.randn(512, 512).to(device="cuda:0")

# import pdb; pdb.set_trace()
# print(dir(new_fn))
# torch._dynamo.export(new_fn, input_tensor1, input_tensor2)

# import logging
# import triton
# import triton.language as tl



# torch._logging.set_logs(dynamo = logging.DEBUG)
# torch._inductor.config.trace.enabled = True
# torch._dynamo.config.verbose = True
# torch._inductor.config.debug = True

# new_testFunc = torch.compile(testFunc, backend="inductor")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        f"testTrace",
    ),
    with_stack=True,
    with_modules=True,) as prof:
    
    for i in range(10):
        # input = torch.randn(1000, ).cuda()
        output = new_fn(input_tensor1, input_tensor2, input_tensor3, input_tensor4)
        # output = fn2(input_tensor1, input_tensor2, input_tensor3, input_tensor4)
        
        if i % 2 == 0:
            prof.step()
        i = i + 1
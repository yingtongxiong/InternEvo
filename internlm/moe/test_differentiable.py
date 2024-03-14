from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

gumbel_map: Dict[torch.device, Callable] = {}

def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)

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

class Model(torch.nn.Module):
    
    def forward(self, logits):
        gates = F.softmax(logits, dim=1)
        indecies = torch.argmax(gates, dim=1)
        mask = F.one_hot(indecies, num_classes=logits.shape[1])
        output = torch.einsum("se,se->se", gates, mask)
        
        return output

class ModelWithoutOneHot(torch.nn.Module):
    
    def forward(self, logits, mask):
        gates = F.softmax(logits, dim=1)
        output = torch.einsum("se,se->se", gates, mask)
        return output

def test1():
    device = torch.device("cuda:0")
    logits = torch.randn((3, 4), device=device, requires_grad=True)
    with torch.no_grad():
        gates = F.softmax(logits, dim=1)
        indecies = torch.argmax(gates, dim=1)
        mask = F.one_hot(indecies, num_classes=logits.shape[1])
        logits_wo = logits.clone()
    
    logits_wo.requires_grad = True
    
    model = Model()
    model_wo = ModelWithoutOneHot()
    output = model(logits)
    output_wo = model_wo(logits_wo, mask)
    
    loss = torch.sum(output)
    loss.backward()
    
    loss_wo = torch.sum(output_wo)
    loss_wo.backward()

class ModelTop2Gate(torch.nn.Module):
    
    def forward(self, logits, noise, capacity):
        gates = F.softmax(logits, dim=1)
        # Create a mask for 1st's expert per token
        indices1_s = torch.argmax(gates, dim=1)
        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        # noise = gumbel_rsample(logits.shape, device=logits.device)
        logits_w_noise = logits + noise
        # Replace top-expert with min value
        logits_except1 = logits_w_noise.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
        indices2_s = torch.argmax(logits_except1, dim=1)
        mask2 = F.one_hot(indices2_s, num_classes=num_experts)

        # Compute locations in capacity buffer
        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations2 = torch.cumsum(mask2, dim=0) - 1
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

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
        locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
        locations2_sc = F.one_hot(locations2_s, num_classes=capacity).type_as(logits)
        combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        dispatch_mask = combine_weights.bool()

        return l_aux, combine_weights, dispatch_mask


class ModelTop2GateWithoutSample(torch.nn.Module):
    
    def forward(self, logits, noise, capacity):
        gates = F.softmax(logits, dim=1)
        # Create a mask for 1st's expert per token
        indices1_s = torch.argmax(gates, dim=1)
        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits_w_noise = logits + noise
        # Replace top-expert with min value
        logits_except1 = logits_w_noise.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
        indices2_s = torch.argmax(logits_except1, dim=1)
        mask2 = F.one_hot(indices2_s, num_classes=num_experts)

        # Compute locations in capacity buffer
        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations2 = torch.cumsum(mask2, dim=0) - 1
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

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
        locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
        locations2_sc = F.one_hot(locations2_s, num_classes=capacity).type_as(logits)
        combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        dispatch_mask = combine_weights.bool()

        return l_aux, combine_weights, dispatch_mask

class ModelTop2GateWithoutOneHot(torch.nn.Module):
    
    def forward(self, logits, mask1, mask2, locations1_sc, locations2_sc):
        gates = F.softmax(logits, dim=1)
        num_experts = int(gates.shape[1])

        # Compute l_aux
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.type_as(logits), dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts

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

class ModelFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: torch.Any, logits, mask1, mask2, locations1_sc, locations2_sc) -> torch.Any:
        gates = F.softmax(logits, dim=1)
        num_experts = int(gates.shape[1])
        mask1_float = mask1.type_as(logits)
        mask2_float = mask2.type_as(logits)
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1_float, dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts
        gates1_s = torch.einsum("se,se->s", gates, mask1_float)
        gates2_s = torch.einsum("se,se->s", gates, mask2_float)
        denom_s = gates1_s + gates2_s
        ctx.save_for_backward(me, ce, locations1_sc, locations2_sc, mask1_float, mask2_float, denom_s, gates1_s, gates2_s, gates)
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s = gates1_s / denom_s
        gates2_s = gates2_s / denom_s
        gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
        gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
        combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        dispatch_mask = combine_weights.bool()
        
        ctx.gates_row = gates.shape[0]
        ctx.gates_col = gates.shape[1]
        ctx.num_experts = num_experts
        
        return l_aux, combine_weights, dispatch_mask
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        me, ce, locations1_sc, locations2_sc, mask1_float, mask2_float, denom_s, gates1_s, gates2_s, gates = ctx.saved_tensors
        grad_gates1 = torch.einsum("sec,sc->se", grad_outputs[1], locations1_sc)
        grad_gates2 = torch.einsum("sec,sc->se", grad_outputs[1], locations2_sc)
        
        grad_gates1_s = torch.einsum("se,se->s", grad_gates1, mask1_float)
        grad_gates2_s = torch.einsum("se,se->s", grad_gates2, mask2_float)
        
        denom_s_output = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        
        grad_gates1_s_ = grad_gates1_s / denom_s_output
        grad_gates2_s_ = grad_gates2_s / denom_s_output
        grad_denom_s = -(grad_gates1_s * gates1_s + grad_gates2_s * gates2_s) / (denom_s ** 2)
        grad_denom_s[denom_s < torch.finfo(denom_s.dtype).eps] = 0
        
        grad_gates1_s_final = grad_gates1_s_ + grad_denom_s
        grad_gates2_s_final = grad_gates2_s_ + grad_denom_s
        
        grad_gates1 = torch.einsum("s,se->se", grad_gates1_s_final, mask1_float)
        grad_gates2 = torch.einsum("s,se->se", grad_gates2_s_final, mask2_float)
        
        gates_row = ctx.gates_row
        gates_col = ctx.gates_col
        num_experts = ctx.num_experts
        size = ce.shape[0]
        grad_me = grad_outputs[0] * num_experts * num_experts * ce / size
        grad_gates3 = grad_me / gates_row * torch.ones((gates_row, gates_col), device=me.device)
        
        grad_gates = grad_gates1 + grad_gates2 + grad_gates3
              
        grad_logits = []
        for i in range(grad_gates.shape[0]):
            softmax_grad = torch.diag(gates[i]) - torch.ger(gates[i], gates[i])
            grad_logits.append(torch.matmul(softmax_grad, grad_gates[i].t()))
        grad_logits = torch.stack(grad_logits)
        
        return grad_logits, None, None, None, None
        

def testGating():
    device = torch.device("cuda:0")
    logits = torch.randn((4096, 12), device=device, requires_grad=True)
    capacity = _capacity(logits, torch.tensor(2), torch.tensor(3))
    noise = gumbel_rsample(logits.shape, device=logits.device)
    
    with torch.no_grad():
        logits_wo = logits.clone()
        logits_autograd = logits.clone()
        gates = F.softmax(logits_wo, dim=1)
        indices1_s = torch.argmax(gates, dim=1)
        mask1 = F.one_hot(indices1_s, num_classes=logits.shape[1])
        logits_w_noise = logits + noise
        logits_except1 = logits_w_noise.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
        
        indices2_s = torch.argmax(logits_except1, dim=1)
        
        mask2 = F.one_hot(indices2_s, num_classes=logits.shape[1])
        locations1 = torch.cumsum(mask1, dim=0) - 1
        locations2 = torch.cumsum(mask2, dim=0) - 1
        locations2 += torch.sum(mask1, dim=0, keepdim=True)
        
        mask1 *= torch.lt(locations1, capacity)
        mask2 *= torch.lt(locations2, capacity)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        locations2_s = torch.sum(locations2 * mask2, dim=1)
        locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
        locations2_sc = F.one_hot(locations2_s, num_classes=capacity).type_as(logits)
        
    
    logits_wo.requires_grad = True
    logits_autograd.requires_grad = True
    
    model = ModelTop2Gate()
    model_wo = ModelTop2GateWithoutOneHot()
    model_autograd = ModelFunction.apply
    output1, output2, output3 = model(logits, noise, capacity)
    output1wo, output2wo, output3wo = model_wo(logits_wo, mask1, mask2, locations1_sc, locations2_sc)
    output1_ag, output2_ag, output3_ag = model_autograd(logits_autograd, mask1, mask2, locations1_sc, locations2_sc)
    
    assert torch.allclose(output1, output1wo)
    assert torch.allclose(output2, output2wo)
    assert torch.allclose(output3, output3wo)
    
    assert torch.allclose(output1, output1_ag)
    assert torch.allclose(output2, output2_ag)
    assert torch.allclose(output3, output3_ag)
    
    loss = torch.sum(torch.sum(output1 * (output2 + output3)))
    loss.backward()
    
    loss_wo = torch.sum(torch.sum(output1wo * (output2wo + output3wo)))
    loss_wo.backward()
    
    loss_ag = torch.sum(torch.sum(output1_ag * (output2_ag + output3_ag)))
    loss_ag.backward()

    assert torch.allclose(logits.grad, logits_wo.grad)
    assert torch.allclose(logits.grad, logits_autograd.grad, atol=1e-4)
    print()

def testSample():
    device = torch.device("cuda:0")
    logits = torch.randn((3, 4), device=device, requires_grad=True)
    capacity = _capacity(logits, torch.tensor(2), torch.tensor(3))
    
    with torch.no_grad():
        logits_wo = logits.clone()
    
    logits_wo.requires_grad = True
    
    model = ModelTop2Gate()
    noise, output1, output2, output3 = model(logits, capacity)
    
    with torch.no_grad():
        noise_wo = noise.clone()
    
    model_wo = ModelTop2GateWithoutSample()
    output1wo, output2wo, output3wo = model_wo(logits_wo, noise_wo, capacity)
    
    loss = torch.sum(torch.sum(output1 * (output2 + output3)))
    loss_wo = torch.sum(torch.sum(output1wo * (output2wo + output3wo)))
    
    loss.backward()
    loss_wo.backward()
    
    import pdb; pdb.set_trace()
    print()

# test1()
testGating()
# testSample()
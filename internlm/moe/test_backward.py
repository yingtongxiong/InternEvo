import torch
import torch.nn.functional as F

class TorchModel(torch.nn.Module):
    
    def forward(self, gates1_s, locations1_sc, gates2_s, locations2_sc, mask1_float, mask2_float):
        gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
        gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
        combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
        output = combine1_sec + combine2_sec
        return output

class MyModel(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, gates1_s, locations1_sc, gates2_s, locations2_sc, mask1_float, mask2_float):
        gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
        gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
        combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
        output = combine1_sec + combine2_sec
        ctx.save_for_backward(gates1_s, locations1_sc, gates2_s, locations2_sc, mask1_float, mask2_float, gates1, gates2)
        return output
    
    @staticmethod
    def backward(ctx, grad_output) -> torch.Any:
        gates1_s, locations1_sc, gates2_s, locations2_sc, mask1_float, mask2_float, gates1, gates2 = ctx.saved_tensors
        
        gates1_grad = torch.einsum("sec,sc->se", grad_output, locations1_sc)
        locations1_sc_grad = torch.einsum("sec,se->sc", grad_output, gates1)
        
        gates2_grad = torch.einsum("sec,sc->se", grad_output, locations2_sc)
        locations2_sc_grad = torch.einsum("sec,se->sc", grad_output, gates2)
        
        gates1_s_grad = torch.einsum("se,se->s", gates1_grad, mask1_float)
        mask1_float_grad = torch.einsum("se,s->se", gates1_grad, gates1_s)
        
        gates2_s_grad = torch.einsum("se,se->s", gates2_grad, mask2_float)
        mask2_float_grad = torch.einsum("se,s->se", gates2_grad, gates2_s)
        
        return gates1_s_grad, locations1_sc_grad, gates2_s_grad, locations2_sc_grad, mask1_float_grad, mask2_float_grad

def test():
    device = torch.device("cuda:0")
    s, e, c = 2, 2, 3
    
    gates1_s_ref = torch.randn((s,), device=device, requires_grad=True)
    locations1_sc_ref = torch.randn((s, c), device=device, requires_grad=True)
    gates2_s_ref = torch.randn((s,), device=device, requires_grad=True)
    locations2_sc_ref = torch.randn((s, c), device=device, requires_grad=True)
    mask1_float_ref = torch.randn((s, e), device=device, requires_grad=True)
    mask2_float_ref = torch.randn((s, e), device=device, requires_grad=True)
    
    with torch.no_grad():
        gates1 = gates1_s_ref.clone()
        locations1_sc = locations1_sc_ref.clone()
        gates2 = gates2_s_ref.clone()
        locations2_sc = locations2_sc_ref.clone()
        mask1_float = mask1_float_ref.clone()
        mask2_float = mask2_float_ref.clone()
        
    gates1.requires_grad = True
    locations1_sc.requires_grad = True
    gates2.requires_grad = True
    locations2_sc.requires_grad = True
    mask1_float.requires_grad = True
    mask2_float.requires_grad = True
    
    model_ref = TorchModel()
    model = MyModel.apply
    
    output_ref = model_ref(gates1_s_ref, locations1_sc_ref, gates2_s_ref, locations2_sc_ref, mask1_float_ref, mask2_float_ref)
    output = model(gates1, locations1_sc, gates2, locations2_sc, mask1_float, mask2_float)
    
    assert torch.allclose(output_ref, output)
    
    loss_ref = torch.sum(torch.sum(torch.sum(output_ref)))
    loss = torch.sum(torch.sum(torch.sum(output)))
    
    loss_ref.backward()
    loss.backward()

    assert torch.allclose(gates1.grad, gates1_s_ref.grad)
    assert torch.allclose(locations1_sc.grad, locations1_sc_ref.grad)
    assert torch.allclose(gates2.grad, gates2_s_ref.grad)
    assert torch.allclose(locations2_sc.grad, locations2_sc_ref.grad)
    assert torch.allclose(mask1_float.grad, mask1_float_ref.grad)
    assert torch.allclose(mask2_float.grad, mask2_float_ref.grad)
        

class BaseLauxModel(torch.nn.Module):
    
    def forward(self, logits, mask1_float, mask2_float, locations1_sc, locations2_sc) -> torch.Any:
        gates = F.softmax(logits, dim=1)
        num_experts = int(gates.shape[1])
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1_float, dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts
        gates1_s = torch.einsum("se,se->s", gates, mask1_float)
        gates2_s = torch.einsum("se,se->s", gates, mask2_float)
        denom_s = gates1_s + gates2_s
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s = gates1_s / denom_s
        gates2_s = gates2_s / denom_s
        gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
        gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
        combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        return l_aux, combine_weights


class LauxModel(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: torch.Any, logits, mask1_float, mask2_float, locations1_sc, locations2_sc) -> torch.Any:
        gates = F.softmax(logits, dim=1)
        num_experts = int(gates.shape[1])
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
        
        ctx.gates_row = gates.shape[0]
        ctx.gates_col = gates.shape[1]
        ctx.num_experts = num_experts
        
        return l_aux, combine_weights
    
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
        
        # me, ce = ctx.saved_tensors
        # gates_row = ctx.gates_row
        # gates_col = ctx.gates_col
        # num_experts = ctx.num_experts
        # size = ce.shape[0]
        # grad_me = grad_output * num_experts * num_experts * ce / size
        # grad_me = grad_me / gates_row * torch.ones((gates_row, gates_col), device=me.device)
        
        # return grad_me, None, None


def test():
    device = torch.device("cuda:0")
    s, e, c = 4 * 1024, 4, 3
    gates1_s = torch.randn((s, ), device=device, requires_grad=True)
    gates2_s = torch.randn((s, ), device=device, requires_grad=True)
    mask1_float = torch.randn((s, e), device=device)
    mask2_float = torch.randn((s, e), device=device)
    locations1_sc = torch.randn((s, c), device=device)
    locations2_sc = torch.randn((s, c), device=device)
    denom_s = torch.randn((s,), device=device, requires_grad=True)
    logits = torch.randn((s, e), device=device, requires_grad=True)
    
    with torch.no_grad():
        gates1_s_ = gates1_s.clone()
        gates2_s_ = gates2_s.clone()
        mask1_float_ = mask1_float.clone()
        mask2_float_ = mask2_float.clone()
        locations1_sc_ = locations1_sc.clone()
        locations2_sc_ = locations2_sc.clone()
        denom_s_ = denom_s.clone()
        logits_ = logits.clone()
    
    gates1_s_.requires_grad = True
    gates2_s_.requires_grad = True
    denom_s_.requires_grad = True
    logits_.requires_grad = True
    
    model = BaseLauxModel()
    model_ = LauxModel.apply
    
    output = model(logits, mask1_float, mask2_float, locations1_sc, locations2_sc)
    output_ = model_(logits_, mask1_float_, mask2_float_, locations1_sc_, locations2_sc_)
    
    assert torch.allclose(output[0], output_[0])
    assert torch.allclose(output[1], output_[1])
    
    loss = torch.sum(torch.mean(output[1] * output[0]))
    loss_ = torch.sum(torch.mean(output_[1] * output_[0]))
    
    loss.backward()
    loss_.backward()
    
    # import pdb; pdb.set_trace()
    assert torch.allclose(logits.grad, logits_.grad, atol=1e-4)
    # assert torch.allclose(gates1_s.grad, gates1_s_.grad, atol=1e-4)
    # assert torch.allclose(gates2_s.grad, gates2_s_.grad, atol=1e-4)
    # assert torch.allclose(denom_s.grad, denom_s_.grad, atol=1e-4)

test()
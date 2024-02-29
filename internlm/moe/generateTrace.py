import torch
import torch.nn.functional as F

from testTriton import fused_dim1

def softmax_argmax(input: torch.Tensor):
    softmax_output = F.softmax(input, dim=1)
    argmax_output = torch.argmax(softmax_output, dim=1)
    return argmax_output

iteration = 10

with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("test",
        ),
        with_stack=True,
        with_modules=True,
        profile_memory=True,
    ) as prof:
    
        device = torch.device('cuda:0')
        shape = (4096, 4096)
        input = torch.randn(shape, device=device)
        
        for i in range(0, iteration):
            # output = softmax_argmax(input)
            output = fused_dim1(input)
            
            if i % 2 == 0:
                prof.step()

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    
    tl.device_print("device pid, offsets, x, y ", pid, offsets, x, y)
    # tl.device_print(" offset = ", offsets)
    # tl.device_print(" x = ", x)
    # tl.device_print(" y = ", y)
    # tl.device_print("===============")
    
    tl.store(output_ptr + offsets, output, mask= mask)

def add(x: torch.Tensor, y: torch.Tensor):
    
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
    
    return output


@triton.jit
def sum_dim1_kernel(
    a_ptr, c_ptr,
    stride_a0, stride_a1,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offset_a = pid * BLOCK_SIZE * stride_a0
    
    v = tl.arange(0, BLOCK_SIZE)
    w = tl.arange(0, N)
    a_ptrs = a_ptr + offset_a + v[:, None] * stride_a0 + w[None, :] * stride_a1
    
    ad = tl.load(a_ptrs)
    a = tl.sum(ad, axis=1)
    
    c_ptrs = c_ptr + pid * BLOCK_SIZE + v
    tl.store(c_ptrs, a)
    
    tl.device_print("pid, offset_a, a, sum, c_ptrs: ", pid, offset_a, ad, a, c_ptrs)


@triton.jit
def _sum_dim0_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    # stride_output_row,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # 指处理哪一列
    offset_col = pid * BLOCK_SIZE * stride_input_col
    
    row = tl.arange(0, M)
    col = tl.arange(0, BLOCK_SIZE)
    
    # input_ptrs = input_ptr + offset_col + row[None, :] * stride_input_row + col[:, None] * stride_input_col
    input_ptrs = input_ptr + offset_col + row[:, None] * stride_input_row + col[None, :] * stride_input_col
    
    input_data = tl.load(input_ptrs)
    # input_sum = tl.sum(input_data, axis=1)
    input_sum = tl.sum(input_data, axis=0)
    
    output_ptrs = output_ptr + pid * BLOCK_SIZE + col
    tl.device_print("pid, input_data, row, col, input_sum: ", pid, input_data, row[:, None] * stride_input_row, col[None, :] * stride_input_col, input_sum)
    tl.store(output_ptrs, input_sum)

def sum_dim0():
    device = torch.device('cuda', 0)
    a = torch.arange(0, 8).reshape(2, 4).to(device=device)
    # c = torch.zeros((1, 4), device=device)
    c = torch.zeros((4,), device=device)
    
    print(f"input = {a}")
    
    stride_a0, stride_a1 = a.stride()
    # stride_c0, stride_c1 = c.stride()
    M, N = a.shape
    
    grid = lambda meta: (2, )
    _sum_dim0_kernel[grid](
        a, c,
        stride_a0, stride_a1,
        # stride_c0,
        M, N,
        BLOCK_SIZE=2
    )
    
    expect_res = torch.sum(a, axis=0, keepdim=True)
    
    print(f"sum kernel output = {c}")
    print(f"expected: {expect_res.shape}, {expect_res}")
    

def sum_dim1():
    device = torch.device('cuda', 0)
    a = torch.arange(0, 8).reshape(2, 4).to(device=device)
    c = torch.zeros((2, 1), device=device)
    
    print(f"input = {a}")
    
    stride_a0, stride_a1 = a.stride()
    M, N = a.shape
    
    grid = lambda meta: (2,)
    sum_dim1_kernel[grid](
        a, c,
        stride_a0, stride_a1,
        M, N,
        BLOCK_SIZE=4,
    )
    
    expect_res = torch.sum(a, axis=1, keepdim=True)
    
    print(f"sum kernel output = {c}")
    print(f"expected: {expect_res.shape}, {expect_res}")


@triton.jit
def _sum2d_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    dim: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # obtain the block id/program id
    pid = tl.program_id(axis=0)
    
    
    if dim == 0:
        # sum the data along the row dimension
        offset_input = pid * BLOCK_SIZE * stride_input_col
        row = tl.arange(0, M)
        col = tl.arange(0, BLOCK_SIZE)
    else:
        # sum the data along the column dimension
        offset_input = pid * BLOCK_SIZE * stride_input_row
        row = tl.arange(0, BLOCK_SIZE)
        col = tl.arange(0, N)
    
    # obtain the input data pointer according to the pid
    input_ptrs = input_ptr + offset_input + row[:, None] * stride_input_row + col[None, :] * stride_input_col
    # load data
    input_data = tl.load(input_ptrs)
    # conduct the sum
    input_sum = tl.sum(input_data, axis=dim)
    
    # calculate the output pointer
    output_ptrs = output_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # store the result to the output
    tl.store(output_ptrs, input_sum)

def sum2d(input: torch.Tensor, dim: int = 0, keepdim: bool = False):
    
    M, N = input.shape
    stride_row, stride_col = input.stride()
    
    # calculate the output shape
    if keepdim == True:
        shape = (M, 1) if dim == 1 else (1, N)
    else:
        shape = (M,) if dim == 1 else (N, )
    
    output = torch.zeros(shape, device=input.device, dtype=input.dtype)
    
    BLOCK_SIZE = 1
    GRID_SIZE = M if dim == 1 else N
    grid = lambda meta: (triton.cdiv(GRID_SIZE, BLOCK_SIZE), )
    _sum2d_kernel[grid](
        input,
        output,
        stride_row, stride_col,
        dim=dim,
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def benchmark_sum():
    
    atol = 1e-5
    
    device = torch.device('cuda:0')
    input = torch.randn((256, 256)).to(device=device)
    
    # dim = 0, keepdim = True
    output1_triton = sum2d(input, dim=0, keepdim=True)
    output1_torch = torch.sum(input, dim=0, keepdim=True)
    assert output1_triton.shape == output1_torch.shape
    if not torch.allclose(output1_triton, output1_torch, atol=atol):
        print(f"the difference is : {output1_triton - output1_torch}")
    
    # dim = 0, keepdim = False
    output2_triton = sum2d(input, dim=0, keepdim=False)
    output2_torch = torch.sum(input, dim=0, keepdim=False)
    assert output2_triton.shape == output2_torch.shape
    if not torch.allclose(output2_triton, output2_torch, atol=atol):
        print(f"the difference is : {output2_triton - output2_torch}")
    
    # dim = 1, keepdim = True
    output3_triton = sum2d(input, dim=1, keepdim=True)
    output3_torch = torch.sum(input, dim=1, keepdim=True)
    assert output3_triton.shape == output3_torch.shape
    if not torch.allclose(output3_triton, output3_torch, atol=atol):
        print(f"the difference is : {output3_triton - output3_torch}")
    
    # dim = 1, keepdim = False
    output4_triton = sum2d(input, dim=1, keepdim=False)
    output4_torch = torch.sum(input, dim=1, keepdim=False)
    assert output4_triton.shape == output4_torch.shape
    if not torch.allclose(output4_triton, output4_torch, atol=atol):
        print(f"the difference is : {output4_triton - output4_torch}")

@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    dim: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    #TODO this kernel only suport the BLOCK_SIZE=1
    # each block process BLOCK_SIZE rows or columns
    
    # obtain the block id in grid
    pid = tl.program_id(axis=0)
    
    if dim == 0:
        offset = pid * BLOCK_SIZE * stride_input_col
        # the row range
        row = tl.arange(0, M)
        # the col range
        col = tl.arange(0, BLOCK_SIZE)
    else: # dim == 1
        # compute the offset from the origin input_ptr
        offset = pid * BLOCK_SIZE * stride_input_row
        # the row range
        row = tl.arange(0, BLOCK_SIZE)
        # the col range
        col = tl.arange(0, N)
    
    # obtain the actual start input pointer
    input_ptrs = input_ptr + offset + row[:, None] * stride_input_row + col[None, :] * stride_input_col
    
    # load data
    input_data = tl.load(input_ptrs)
    
    # compute the exp
    input_exp = tl.exp(input_data)
    # compute the denominator
    denorm = tl.sum(input_exp, axis=dim)
    # compute the normalize value
    # tl.device_print("pid, input_data, input_exp, denorm: ", pid, input_data, input_exp, denorm)
    input_norm = input_exp / denorm
    
    # compute the output location
    output_ptrs = output_ptr + offset + row[:, None] * stride_input_row + col[None, :] * stride_input_col
    # store the output value
    tl.store(output_ptrs, input_norm) 

@triton.jit
def _softmax_kernel2(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    dim: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # each block process BLOCK_SIZE elements
    
    pid = tl.program_id(axis=0)
    
    if dim == 0:
        col = pid
        row = tl.arange(0, BLOCK_SIZE)
        # input_ptrs = input_ptr + row[:, None] * stride_input_row + col * stride_input_col
        # output_ptrs = output_ptr + row[:, None] * stride_input_row + col * stride_input_col
        input_ptrs = input_ptr + row * stride_input_row + col * stride_input_col
        output_ptrs = output_ptr + row * stride_input_row + col * stride_input_col
    else:
        col = tl.arange(0, BLOCK_SIZE)
        row = pid
        # input_ptrs = input_ptr + row * stride_input_row + col[None, :] * stride_input_col
        # output_ptrs = output_ptr + row * stride_input_row + col[None, :] * stride_input_col
        input_ptrs = input_ptr + row * stride_input_row + col * stride_input_col
        output_ptrs = output_ptr + row * stride_input_row + col * stride_input_col
    
    # input_ptrs = input_ptr + row[:, None] * stride_input_row + col[None, :] * stride_input_col
    
    # load data
    input_data = tl.load(input_ptrs)
    
    input_exp = tl.exp(input_data)
    denorm = tl.sum(input_exp, axis=dim)
    input_norm = input_exp / denorm
    
    # compute the output 
    # output_ptrs = output_ptr + row[:, None] * stride_input_row + col[None, :] * stride_input_col
    # store output
    tl.store(output_ptrs, input_norm)

def softmax2d(input: torch.Tensor, dim: int = 0):
    output = torch.zeros(input.shape, dtype=input.dtype, device=input.device)
    
    M, N = input.shape
    stride_row, stride_col = input.stride()
    
    # BLOCK_SIZE = 2
    # GRID_SIZE = M if dim == 1 else N
    
    # grid = lambda meta: (triton.cdiv(GRID_SIZE, BLOCK_SIZE), )
    # _softmax_kernel[grid](
    #     input,
    #     output,
    #     stride_row, stride_col,
    #     dim=dim,
    #     M=M, N=N,
    #     BLOCK_SIZE=BLOCK_SIZE,
    # )
    
    block_size = N if dim == 1 else M
    grid_size = M if dim == 1 else N
    BLOCK_SIZE = triton.next_power_of_2(block_size)
    
    _softmax_kernel2[(grid_size, )](
        input,
        output,
        stride_row, stride_col,
        dim=dim,
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def benchmark_softmax():
    atol = 1e-8
    
    device = torch.device("cuda:0")
    shape = (4096, 4096)
    input = torch.randn(shape, device=device)
    # input = torch.arange(0, 16).reshape(shape).to(torch.float32).to(device)
    print(f"input = {input}")
    
    # dim == 0
    output1_triton = softmax2d(input, dim=0)
    output1_torch = F.softmax(input, dim=0)
    assert output1_triton.shape == output1_torch.shape
    assert torch.allclose(output1_triton, output1_torch, atol=atol)
    
    # dim == 1
    output2_triton = softmax2d(input, dim=1)
    output2_torch = F.softmax(input, dim=1)
    assert output2_triton.shape == output2_torch.shape
    assert torch.allclose(output2_triton, output2_torch, atol=atol)
    
    print("Success!")

@triton.jit
def _argmax_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    col = tl.arange(0, BLOCK_SIZE)
    
    input_ptrs = input_ptr + row * stride_input_row + col * stride_input_col
    
    input_data = tl.load(input_ptrs, mask=row < M, other=-float('inf'))
    input_argmax = tl.argmax(input_data, axis=0)
    
    output_ptrs = output_ptr + row 
    
    tl.store(output_ptrs, input_argmax)

def argmax():
    device = torch.device("cuda:0")
    
    # input = torch.arange(0, 8).reshape(2, 4).to(device)
    # input[0][0] = 8
    input = torch.randn((2048, 4096), device=device)
    output = torch.zeros((2048, ), dtype=input.dtype, device=input.device)
    M, N = input.shape
    stride0, stride1 = input.stride()
    
    BLOCK_SIZE = N
    # import pdb; pdb.set_trace()
    _argmax_kernel[(M, )](
        input,
        output,
        stride0, stride1,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    print(f"output = {output}, dtype={output.dtype}")
    torch_output = torch.argmax(input, dim=1)
    print(f"torch output = {torch_output}, dtype={torch_output.dtype}")
    
    assert torch.allclose(output.to(torch.int64), torch_output)
    
@triton.jit
def _softmax_dim1_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # the softmax computation for each row is independent
    # each block process each row
    row_id = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    
    input_ptrs = input_ptr + row_id * stride_input_row + col_offset * stride_input_col
    
    # load data
    input_data = tl.load(input_ptrs, mask=col_offset < N, other=-float("inf"))
    input_exp = tl.exp(input_data)
    denom = tl.sum(input_exp, axis=0)
    softmax_output = input_exp / denom
    
    # compute the output pointer
    output_ptrs = output_ptr + row_id * stride_output_row + col_offset * stride_output_col
    # store data
    tl.store(output_ptrs, softmax_output, mask=col_offset < N)

def softmax_dim1(input: torch.Tensor):
    M, N = input.shape
    stride_input_row, stride_input_col = input.stride()
    
    output = torch.zeros(input.shape, device=input.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    _softmax_dim1_kernel[(M, )](
        input,
        output,
        stride_input_row, stride_input_col,
        stride_input_row, stride_input_col,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def benchmark_softmax_dim1():
    device = torch.device("cuda:0")
    shape = (4096, 4096)
    
    input = torch.randn(shape, device=device)
    
    triton_output = softmax_dim1(input)
    torch_output = F.softmax(input, dim=1)
    
    assert triton_output.shape == torch_output.shape
    assert torch.allclose(triton_output, torch_output)

@triton.jit
def _fused_dim1_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # the softmax computation for each row is independent
    # each block process each row
    row_id = tl.program_id(axis=0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    
    input_ptrs = input_ptr + row_id * stride_input_row + col_offset * stride_input_col
    
    # load data
    input_data = tl.load(input_ptrs, mask=col_offset < N, other=-float("inf"))
    input_exp = tl.exp(input_data)
    denom = tl.sum(input_exp, axis=0)
    softmax_output = input_exp / denom
    argmax_output = tl.argmax(softmax_output, axis=0)
    
    # compute the output pointer
    # output_ptrs = output_ptr + row_id
    output_ptrs = output_ptr + row_id * stride_output_row + argmax_output * stride_output_col
    # store data
    # tl.store(output_ptrs, argmax_output)
    tl.store(output_ptrs, 1, mask=argmax_output < N)


def fused_dim1(input: torch.Tensor):

    M, N = input.shape
    stride_row, stride_col = input.stride()
 
    output = torch.zeros((M, N), dtype=torch.int64, device=input.device)
    
    stride_output_row, stride_output_col = output.stride()
    
    BLOCK_SIZE = triton.next_power_of_2(N)

    _fused_dim1_kernel[(M, )](
        input,
        output,
        stride_row, stride_col,
        stride_output_row, stride_output_col,
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

@triton.jit
def _fused_dim0_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # dim = 0:
    # the softmax computation for each column is independent
    # each block processes each column
    col_id = tl.program_id(axis=0)
    row_offset = tl.arange(0, BLOCK_SIZE)
    
    input_ptrs = input_ptr + row_offset * stride_input_row + col_id * stride_input_col
    
    # load data
    input_data = tl.load(input_ptrs, mask=row_offset < M, other=-float("inf"))
    input_exp = tl.exp(input_data)
    denom = tl.sum(input_exp, axis=0)
    softmax_output = input_exp / denom
    argmax_output = tl.argmax(softmax_output, axis=0)
    
    # compute the output pointer
    # output_ptrs = output_ptr + col_id
    output_ptrs = output_ptr + col_id * stride_output_row + argmax_output * stride_output_col
    # store data
    # tl.store(output_ptrs, argmax_output)
    tl.store(output_ptrs, 1, mask=argmax_output < M)

def fused_dim0(input: torch.Tensor):
    
    M, N = input.shape
    stride_row, stride_col = input.stride()
 
    output = torch.zeros((N, M), dtype=torch.int64, device=input.device)
    stride_output_row, stride_output_col = output.stride()
    
    BLOCK_SIZE = triton.next_power_of_2(M)

    _fused_dim0_kernel[(N, )](
        input,
        output,
        stride_row, stride_col,
        stride_output_row, stride_output_col,
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def fused2d(input: torch.Tensor, dim: int = 0):
    if dim == 0:
        return fused_dim0(input)
    return fused_dim1(input)

@triton.jit
def _one_hot_kernel(
    input_ptr,
    output_ptr,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # each block process each row 
    row_id = tl.program_id(axis=0)
    
    col_offset = tl.arange(0, BLOCK_SIZE)
    
    input_ptrs = input_ptr + row_id * stride_input_row + col_offset * stride_input_col
    
    input_data = tl.load(input_ptrs, mask=col_offset < N)
    
    input_argmax = tl.argmax(input_data, axis=0)
    
    output_ptrs = output_ptr + row_id * stride_output_row + input_argmax * stride_output_col
    
    tl.store(output_ptrs, 1, mask=input_argmax < N)

def one_hot():
    
    device = torch.device('cuda:0')
    # input = torch.Tensor([[0, 3, 1, 2], [6, 7, 4, 5], [12, 11, 8, 9]]).to(device)
    shape = (4096, 4096)
    input = torch.randn(shape, device=device)
    # output = torch.zeros_like(input).to(torch.int64)
    
    output_triton = fused_dim1(input)
    output_torch = F.one_hot(torch.argmax(F.softmax(input, dim=1), dim=1), num_classes=shape[0])
    
    assert output_triton.shape == output_torch.shape
    assert torch.allclose(output_triton, output_torch)
    
    # stride_row, stride_col = input.stride()
    # M, N = input.shape
    # BLOCK_SIZE = triton.next_power_of_2(N)
    
    # _one_hot_kernel[(M, )](
    #     input,
    #     output,
    #     stride_row, stride_col,
    #     stride_row, stride_col,
    #     M, N,
    #     BLOCK_SIZE=BLOCK_SIZE,
    # )
    
    # print(f"output = {output}")

def benchmak_fused():
    device = torch.device("cuda:0")
    shape = (8192, 8192)
    input = torch.randn(shape, device=device)
    
    # dim = 0
    output0_triton = fused_dim0(input)
    output0_triton_unify = fused2d(input, dim=0)
    output0_torch = F.one_hot(torch.argmax(F.softmax(input, dim=0), dim=0), num_classes=shape[0])
    assert output0_triton.shape == output0_torch.shape
    assert output0_triton_unify.shape == output0_torch.shape
    assert torch.allclose(output0_triton, output0_torch)
    assert torch.allclose(output0_triton_unify, output0_torch)
    
    # dim = 1
    output1_triton = fused_dim1(input)#.to(torch.int64)
    output1_triton_unify = fused2d(input, dim=1)
    output1_torch = F.one_hot(torch.argmax(F.softmax(input, dim=1), dim=1), num_classes=shape[1])
    assert output1_triton.shape == output1_torch.shape
    assert output1_triton_unify.shape == output1_torch.shape
    assert torch.allclose(output1_triton, output1_torch)
    assert torch.allclose(output1_triton_unify, output1_torch)
    
    
    print("Success!")


@triton.jit
def _fused_kernel1(
    logits1_ptr,
    logits2_ptr,
    mask1_ptr,
    mask2_ptr,
    gates_ptr,
    fill_value,
    stride_logits1_row, stride_logits1_col,
    stride_logits2_row, stride_logits2_col,
    stride_mask1_row, stride_mask1_col,
    stride_mask2_row, stride_mask2_col,
    seq_len: tl.constexpr, expert_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # the softmax computation for each row is independent
    # each block process each row
    pid = tl.program_id(axis=0)
    logits1_col = tl.arange(0, BLOCK_SIZE)
    logits2_col = tl.arange(0, BLOCK_SIZE)
    
    logits1_ptrs = logits1_ptr + pid * stride_logits1_row + logits1_col * stride_logits1_col
    gates_ptrs = gates_ptr + pid * stride_logits1_row + logits1_col * stride_logits1_col
    
    # load data
    logits1_data = tl.load(logits1_ptrs, mask=logits1_col < expert_num, other=-float("inf"))
    logits1_exp = tl.exp(logits1_data)
    denom1 = tl.sum(logits1_exp, axis=0)
    softmax1_output = logits1_exp / denom1
    argmax1_output = tl.argmax(softmax1_output, axis=0)
    tl.store(gates_ptrs, softmax1_output)
    
    logits2_ptrs = logits2_ptr + pid * stride_logits2_row + argmax1_output * stride_logits2_col
    # tl.store(logits2_ptr, fill_value, mask=argmax1_output < expert_num)
    
    logits2_ptrs = logits2_ptr + pid * stride_logits2_row + logits2_col * stride_logits2_col
    logits2_data = tl.load(logits2_ptrs, mask=logits1_col < expert_num and logits1_col != argmax1_output, other=fill_value)
    argmax2_output = tl.argmax(logits2_data, axis=0)
    
    # compute the output pointer
    mask1_ptrs = mask1_ptr + pid * stride_mask1_row + argmax1_output * stride_mask1_col
    mask2_ptrs = mask2_ptr + pid * stride_mask2_row + argmax2_output * stride_mask2_col
    # store data
    # tl.store(output_ptrs, argmax_output)
    tl.store(mask1_ptrs, 1, mask=argmax1_output < expert_num)
    tl.store(mask2_ptrs, 1, mask=argmax2_output < expert_num)

def fused_gating1(logits1, logits2):
    
    mask1 = torch.zeros_like(logits1).to(torch.int64)
    mask2 = torch.zeros_like(logits2).to(torch.int64)
    gates = torch.zeros_like(logits1)
    
    stride_logits1_row, stride_logits1_col = logits1.stride()
    stride_logits2_row, stride_logits2_col = logits2.stride()
    stride_mask1_row, stride_mask1_col = mask1.stride()
    stride_mask2_row, stride_mask2_col = mask2.stride()
    seq_len, expert_num = logits1.shape
    
    block_size = triton.next_power_of_2(expert_num)
    fill_value = torch.finfo(logits1.dtype).min
    
    _fused_kernel1[(seq_len, )](
        logits1,
        logits2,
        mask1,
        mask2,
        gates,
        fill_value,
        stride_logits1_row, stride_logits1_col,
        stride_logits2_row, stride_logits2_col,
        stride_mask1_row, stride_mask1_col,
        stride_mask2_row, stride_mask2_col,
        seq_len, expert_num,
        BLOCK_SIZE=block_size,
    )

    return mask1, mask2, gates


@triton.jit
def _fused_kernel2(
    mask1_ptr,
    mask2_ptr,
    gates_ptr,
    locations1,
    locations2,
    res,
    stride_mask_row, stride_mask_col,
    stride_gates_row, stride_gates_col,
    stride_location_row, stride_location_col,
    capacity: tl.constexpr,
    seq_len: tl.constexpr, expert_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    mask_row = tl.arange(0, BLOCK_SIZE)
    
    mask1_ptrs = mask1_ptr + mask_row * stride_mask_row + pid * stride_mask_col
    mask2_ptrs = mask2_ptr + mask_row * stride_mask_row + pid * stride_mask_col
    gates_ptrs = gates_ptr + mask_row * stride_gates_row + pid * stride_gates_col
    
    mask1_data = tl.load(mask1_ptrs, mask=mask_row < seq_len)
    mask2_data = tl.load(mask2_ptrs, mask=mask_row < seq_len)
    gates_data = tl.load(gates_ptrs, mask=mask_row < seq_len)
    
    mask1_sum = tl.sum(mask1_data, axis=0)
    loca1 = tl.cumsum(mask1_data, axis=0) - 1
    loca2 = tl.cumsum(mask2_data, axis=0) - 1 + mask1_sum
    
    loca1_ptrs = locations1 + mask_row * stride_location_row + pid * stride_location_col
    loca2_ptrs = locations2 + mask_row * stride_location_row + pid * stride_location_col
    
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

def fused_gating2(mask1, mask2, gates, capacity=1):
    loca1 = torch.zeros_like(mask1)
    loca2 = torch.zeros_like(mask2)
    
    stride_mask_row, stride_mask_col = mask1.stride()
    stride_gates_row, stride_gates_col = gates.stride()
    
    seq_len, expert_num = mask1.shape
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    res = torch.zeros((expert_num,)).to(mask1.device)
    
    _fused_kernel2[(expert_num, )](
        mask1,
        mask2,
        gates,
        loca1,
        loca2,
        res,
        stride_mask_row, stride_mask_col,
        stride_gates_row, stride_gates_col,
        stride_mask_row, stride_mask_col,
        capacity=capacity,
        seq_len=seq_len, expert_num=expert_num,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return loca1, loca2, res, mask1, mask2


@triton.jit
def _fused_kernel3(
    input1_ptr,
    input2_ptr,
    output1_ptr,
    output2_ptr,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    capacity: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    input_col = tl.arange(0, BLOCK_SIZE)
    
    input1_ptrs = input1_ptr + pid * stride_input_row + input_col * stride_input_col
    input2_ptrs = input2_ptr + pid * stride_input_row + input_col * stride_input_col
    
    input1_data = tl.load(input1_ptrs, mask=input_col < N)
    input2_data = tl.load(input2_ptrs, mask=input_col < N)
    
    input1_sum = tl.sum(input1_data, axis=0)
    input2_sum = tl.sum(input2_data, axis=0)
    
    output1_ptrs = output1_ptr + pid * stride_output_row + input1_sum * stride_output_col
    output2_ptrs = output2_ptr + pid * stride_output_row + input2_sum * stride_output_col
    
    tl.store(output1_ptrs, 1, mask=input1_sum < capacity)
    tl.store(output2_ptrs, 1, mask=input2_sum < capacity)
    
    # tl.device_print(input1_sum.dtype)
    
    
    # locations1_sc = tl.zeros((capacity, ), dtype=tl.float32)
    # locations2_sc = tl.zeros((capacity, ), dtype=tl.float32)
    
    # locations1_sc += tl.where(input1_sum == 1, 1, 0)
    # locations2_sc += tl.where(input2_sum == 1, 1, 0)
    
    # locations1_sc_ptrs = output1_ptr + pid * stride_output_row + tl.arange(0, capacity)
    # locations2_sc_ptrs = output2_ptr + pid * stride_output_row + tl.arange(0, capacity)
    
    # tl.store(locations1_sc_ptrs, locations1_sc)
    # tl.store(locations2_sc_ptrs, locations2_sc)

def fused_gating3(locations1, locations2, capacity=2):
    row, col = locations1.shape
    stride_input_row, stride_input_col = locations1.stride()
    
    output1 = torch.zeros(row, capacity, device=locations1.device)
    output2 = torch.zeros(row, capacity, device=locations1.device)
    stride_output_row, stride_output_col = output1.stride()
    
    BLOCK_SIZE = triton.next_power_of_2(col)
    _fused_kernel3[(row, )](
        locations1,
        locations2,
        output1,
        output2,
        stride_input_row, stride_input_col,
        stride_output_row, stride_output_col,
        capacity=capacity,
        M=row, N=col,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output1, output2

@triton.jit
def _fused_kernel4(
    gates_ptr,
    mask1_ptr,
    mask2_ptr,
    gates1_ptr,
    gates2_ptr,
    stride_gates_row, stride_gates_col,
    stride_mask_row, stride_mask_col,
    min_value: tl.constexpr,
    s: tl.constexpr, e: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    s_pid = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE)
    
    gates_ptrs = gates_ptr + s_pid * stride_gates_row + e_offset * stride_gates_col
    mask1_ptrs = mask1_ptr + s_pid * stride_mask_row + e_offset * stride_mask_col
    mask2_ptrs = mask2_ptr + s_pid * stride_mask_row + e_offset * stride_mask_col
    
    gates_data = tl.load(gates_ptrs, mask=e_offset < e)
    mask1_data = tl.load(mask1_ptrs, mask=e_offset < e)
    mask2_data = tl.load(mask2_ptrs, mask=e_offset < e)
    
    multi1 = gates_data * mask1_data
    multi2 = gates_data * mask2_data
    multi1_sum = tl.sum(multi1, axis=0)
    multi2_sum = tl.sum(multi2, axis=0)
    add_res = multi1_sum + multi2_sum
    add_res = tl.where(add_res < min_value, min_value, add_res)
    multi1_sum /= add_res
    multi2_sum /= add_res
    
    gates1 = multi1_sum * mask1_data
    gates2 = multi2_sum * mask2_data
    
    tl.store(gates1_ptr + s_pid * stride_gates_row + e_offset, gates1, mask=e_offset < e)
    tl.store(gates2_ptr + s_pid * stride_gates_row + e_offset, gates2, mask=e_offset < e)

def fused_gating4(gates, mask1, mask2):
    s, e = gates.shape
    stride_gates_row, stride_gates_col = gates.stride()
    stride_mask_row, stride_mask_col = mask1.stride()
    
    BLOCK_SIZE = triton.next_power_of_2(e)
    
    # gates1 = torch.zeros((s,), device=gates.device)
    # gates2 = torch.zeros((s,), device=gates.device)
    
    gates1 = torch.zeros_like(gates)
    gates2 = torch.zeros_like(gates)
    
    _fused_kernel4[(s, )](
        gates,
        mask1,
        mask2,
        gates1,
        gates2,
        stride_gates_row, stride_gates_col,
        stride_mask_row, stride_mask_col,
        torch.finfo(gates.dtype).eps,
        s=s, e=e,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return gates1, gates2

@triton.jit
def _fused_kernel5(
    gates_ptr,
    mask1_ptr,
    mask2_ptr,
    loca1_ptr,
    loca2_ptr,
    combine_ptr,
    dispatch_mask_ptr,
    stride_gates_s,
    stride_mask_s,
    stride_loca_s,
    stride_combine_s, stride_combine_e, stride_combine_c,
    min_value: tl.constexpr,
    s: tl.constexpr, e: tl.constexpr, c: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr, BLOCK_SIZE_c: tl.constexpr,
):
    
    # s_pid = tl.program_id(axis=0)
    # e_offset = tl.arange(0, BLOCK_SIZE_e)
    # c_offset = tl.arange(0, BLOCK_SIZE_c)
    
    # loca1_ptrs = loca1_ptr + s_pid * stride_loca_s + e_offset
    # loca2_ptrs = loca2_ptr + s_pid * stride_loca_s + e_offset
    # mask1_ptrs = mask1_ptr + s_pid * stride_mask_s + e_offset
    # mask2_ptrs = mask2_ptr + s_pid * stride_mask_s + e_offset
    
    # locations1 = tl.load(loca1_ptrs, mask=e_offset < e)
    # locations2 = tl.load(loca2_ptrs, mask=e_offset < e)
    # mask1_data = tl.load(mask1_ptrs, mask=e_offset < e)
    # mask2_data = tl.load(mask2_ptrs, mask=e_offset < e)
    
    # mask1_data *= tl.where(locations1 < capacity, 1, 0)
    # mask2_data *= tl.where(locations2 < capacity, 1, 0)
    
    # locations1_s = tl.sum(mask1_data * locations1, axis=0)
    # locations2_s = tl.sum(mask2_data * locations2, axis=0)
    # # tl.device_print(locations1_s.shape)
    
    # locations1_sc = tl.zeros((BLOCK_SIZE_c, ), dtype=tl.float32)
    # locations2_sc = tl.zeros((BLOCK_SIZE_c, ), dtype=tl.float32)
    
    
    # locations1_sc += tl.where(locations1_s == 1, 1, 0)
    # locations2_sc += tl.where(locations2_s == 1, 1, 0)
    
    # locations1_sc_ptrs = locations1_ptr + s_pid * stride_locations_s + c_offset
    # locations2_sc_ptrs = locations2_ptr + s_pid * stride_locations_s + c_offset
    
    # tl.store(locations1_sc_ptrs, locations1_sc)
    # tl.store(locations2_sc_ptrs, locations2_sc)
    
    # gates_ptrs = gates_ptr + s_pid * stride_gates_s + e_offset
    
    # gates_data = tl.load(gates_ptrs, mask=e_offset < e)
    
    # multi1 = gates_data * mask1_data
    # multi2 = gates_data * mask2_data
    # multi1_sum = tl.sum(multi1, axis=0)
    # multi2_sum = tl.sum(multi2, axis=0)
    # add_res = multi1_sum + multi2_sum
    # add_res = tl.where(add_res < min_value, min_value, add_res)
    # multi1_sum /= add_res
    # multi2_sum /= add_res
    
    # gates1 = multi1_sum * mask1_data
    # gates2 = multi2_sum * mask2_data

    
    # locations1_sc = tl.expand_dims(locations1_sc, axis=0)
    # locations1_sc = tl.broadcast_to(locations1_sc, (BLOCK_SIZE_e, BLOCK_SIZE_c))
    # gates1 = tl.expand_dims(gates1, axis=1)
    
    # locations2_sc = tl.expand_dims(locations2_sc, axis=0)
    # locations2_sc = tl.broadcast_to(locations2_sc, (BLOCK_SIZE_e, BLOCK_SIZE_c))
    # gates2 = tl.expand_dims(gates2, axis=1)
    
    # combine1 = gates1 * locations1_sc
    # combine2 = gates2 * locations2_sc
    # combine = combine1 + combine2

    # combine_ptrs = combine_ptr + s_pid * stride_combine_s + e_offset[:, None] * stride_combine_e + c_offset[None, :] * stride_combine_c
    
    # mask = (e_offset[:, None] < e) & (c_offset[None, :] < c) 
    
    # tl.store(combine_ptrs, combine, mask=mask)
    
    
    
    
    s_pid = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE_e)
    c_offset = tl.arange(0, BLOCK_SIZE_c)
    
    
    gates_ptrs = gates_ptr + s_pid * stride_gates_s + e_offset
    mask1_ptrs = mask1_ptr + s_pid * stride_mask_s + e_offset
    mask2_ptrs = mask2_ptr + s_pid * stride_mask_s + e_offset
    
    gates_data = tl.load(gates_ptrs, mask=e_offset < e)
    mask1_data = tl.load(mask1_ptrs, mask=e_offset < e)
    mask2_data = tl.load(mask2_ptrs, mask=e_offset < e)
    
    multi1 = gates_data * mask1_data
    multi2 = gates_data * mask2_data
    multi1_sum = tl.sum(multi1, axis=0)
    multi2_sum = tl.sum(multi2, axis=0)
    add_res = multi1_sum + multi2_sum
    add_res = tl.where(add_res < min_value, min_value, add_res)
    multi1_sum /= add_res
    multi2_sum /= add_res
    
    gates1 = multi1_sum * mask1_data
    gates2 = multi2_sum * mask2_data

    
    loca1_ptrs = loca1_ptr + s_pid * stride_loca_s + c_offset
    loca2_ptrs = loca2_ptr + s_pid * stride_loca_s + c_offset
    
    loca1_data = tl.load(loca1_ptrs, mask=c_offset < c)
    loca2_data = tl.load(loca2_ptrs, mask=c_offset < c)
    
    loca1_data = tl.expand_dims(loca1_data, axis=0)
    loca1_data = tl.broadcast_to(loca1_data, (BLOCK_SIZE_e, BLOCK_SIZE_c))
    gates1 = tl.expand_dims(gates1, axis=1)
    
    loca2_data = tl.expand_dims(loca2_data, axis=0)
    loca2_data = tl.broadcast_to(loca2_data, (BLOCK_SIZE_e, BLOCK_SIZE_c))
    gates2 = tl.expand_dims(gates2, axis=1)
    
    combine1 = gates1 * loca1_data
    combine2 = gates2 * loca2_data
    combine = combine1 + combine2

    combine_ptrs = combine_ptr + s_pid * stride_combine_s + e_offset[:, None] * stride_combine_e + c_offset[None, :] * stride_combine_c
    dispatch_mask_ptrs = dispatch_mask_ptr + s_pid * stride_combine_s + e_offset[:, None] * stride_combine_e + c_offset[None, :] * stride_combine_c
    
    mask = (e_offset[:, None] < e) & (c_offset[None, :] < c) 
    
    dispatch_mask_data = tl.where(combine > 0, 1, 0)
    
    tl.store(combine_ptrs, combine, mask=mask)
    tl.store(dispatch_mask_ptrs, dispatch_mask_data, mask=mask)

def fused_gating5(gates, mask1, mask2, loca1, loca2):
    s, e = gates.shape
    s, c = loca1.shape

    # locations1 = torch.zeros((s, c), device=gates.device)
    # locations2 = torch.zeros((s, c), device=gates.device)
    # stride_locations_s, _ = locations1.stride()
    combine = torch.zeros((s,e,c), device=gates.device)
    dispatch_mask = torch.zeros((s,e,c), device=gates.device, dtype=torch.bool)
    
    BLOCK_SIZE_e = triton.next_power_of_2(e)
    BLOCK_SIZE_c = triton.next_power_of_2(c)
    
    stride_gates_s, _ = gates.stride()
    stride_mask_s, _ = mask1.stride()
    stride_loca_s, _ = loca1.stride()
    stride_combine_s, stride_combine_e, stride_combine_c = combine.stride()
    
    _fused_kernel5[(s, )](
        gates,
        mask1,
        mask2,
        loca1,
        loca2,
        combine,
        dispatch_mask,
        stride_gates_s,
        stride_mask_s,
        stride_loca_s,
        stride_combine_s, stride_combine_e, stride_combine_c,
        torch.finfo(gates.dtype).eps,
        s, e, c,
        BLOCK_SIZE_e, BLOCK_SIZE_c,
    )
    
    return combine, dispatch_mask

@triton.jit
def _fused_kernel35(
    gates_ptr,
    input1_ptr, # locations1 * mask1
    input2_ptr, # locations2 * mask2
    mask1_ptr,
    mask2_ptr,
    gates1_ptr,
    gates2_ptr,
    combine_ptr,
    dispatch_mask_ptr,
    stride_gates_s,
    stride_input_s,
    stride_mask_s,
    stride_combine_s, stride_combine_e,
    s: tl.constexpr, e: tl.constexpr, c: tl.constexpr,
    min_value: tl.constexpr,
    BLOCK_SIZE_e: tl.constexpr,
):
    s_pid = tl.program_id(axis=0)
    e_offset = tl.arange(0, BLOCK_SIZE_e)
    
    #locations1_s = torch.sum(locations1 * mask1, dim=1)
    #locations2_s = torch.sum(locations2 * mask2, dim=1)
    
    input1_ptrs = input1_ptr + s_pid * stride_input_s + e_offset
    input2_ptrs = input2_ptr + s_pid * stride_input_s + e_offset
    
    input1_data = tl.load(input1_ptrs, mask=e_offset < e)
    input2_data = tl.load(input2_ptrs, mask=e_offset < e)
    
    locations1_s = tl.sum(input1_data, axis=0)
    locations2_s = tl.sum(input2_data, axis=0)
    
    # output1_ptrs = output1_ptr + pid * stride_output_row + locations1_s * stride_output_col
    # output2_ptrs = output2_ptr + pid * stride_output_row + locations2_s * stride_output_col
    
    # tl.store(output1_ptrs, 1, mask=locations1_s < capacity)
    # tl.store(output2_ptrs, 1, mask=locations2_s < capacity)
    
    gates_ptrs = gates_ptr + s_pid * stride_gates_s + e_offset
    mask1_ptrs = mask1_ptr + s_pid * stride_mask_s + e_offset
    mask2_ptrs = mask2_ptr + s_pid * stride_mask_s + e_offset
    
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
    
    gates1_ptrs = gates1_ptr + s_pid * stride_gates_s + e_offset
    gates2_ptrs = gates2_ptr + s_pid * stride_gates_s + e_offset
    
    tl.store(gates1_ptrs, gates1, mask=e_offset < e)
    tl.store(gates2_ptrs, gates2, mask=e_offset < e)
    # tl.device_print("pid", s_pid, gates1, gates2, gates1 + gates2)

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
    
    # =================================================================================
    # loca1_ptrs = loca1_ptr + s_pid * stride_loca_s + c_offset
    # loca2_ptrs = loca2_ptr + s_pid * stride_loca_s + c_offset
    
    # loca1_data = tl.load(loca1_ptrs, mask=c_offset < c)
    # loca2_data = tl.load(loca2_ptrs, mask=c_offset < c)
    
    # loca1_data = tl.expand_dims(loca1_data, axis=0)
    # loca1_data = tl.broadcast_to(loca1_data, (BLOCK_SIZE_e, BLOCK_SIZE_c))
    # gates1 = tl.expand_dims(gates1, axis=1)
    
    # loca2_data = tl.expand_dims(loca2_data, axis=0)
    # loca2_data = tl.broadcast_to(loca2_data, (BLOCK_SIZE_e, BLOCK_SIZE_c))
    # gates2 = tl.expand_dims(gates2, axis=1)
    
    # combine1 = gates1 * loca1_data
    # combine2 = gates2 * loca2_data
    # combine = combine1 + combine2

    # combine_ptrs = combine_ptr + s_pid * stride_combine_s + e_offset[:, None] * stride_combine_e + c_offset[None, :] * stride_combine_c
    # dispatch_mask_ptrs = dispatch_mask_ptr + s_pid * stride_combine_s + e_offset[:, None] * stride_combine_e + c_offset[None, :] * stride_combine_c
    
    # mask = (e_offset[:, None] < e) & (c_offset[None, :] < c) 
    
    # dispatch_mask_data = tl.where(combine > 0, 1, 0)
    
    # tl.store(combine_ptrs, combine, mask=mask)
    # tl.store(dispatch_mask_ptrs, dispatch_mask_data, mask=mask)

def fused_gating35(gates, input1, input2, mask1, mask2, c=1):
    s, e = gates.shape
    stride_gates_s, _ = gates.stride()
    stride_input_s, _ = input1.stride()
    stride_mask_s, _ = mask1.stride()
    
    gates1 = torch.zeros((s,e), device=gates.device)
    gates2 = torch.zeros((s,e), device=gates.device)
    
    combine = torch.zeros((s, e, c), device=gates.device)
    dispatch_mask = torch.zeros((s, e, c), device=gates.device, dtype=torch.bool)
    
    stride_combine_s, stride_combine_e, stride_combine_c = combine.stride()
    
    min_value = torch.finfo(gates.dtype).eps
    BLOCK_SIZE_e = triton.next_power_of_2(e)
    
    _fused_kernel35[(s, )](
        gates,
        input1,
        input2,
        mask1,
        mask2,
        gates1,
        gates2,
        combine,
        dispatch_mask,
        stride_gates_s,
        stride_input_s,
        stride_mask_s,
        stride_combine_s,
        stride_combine_e,
        s, e, c,
        min_value,
        BLOCK_SIZE_e,
    )
    
    return combine, dispatch_mask
    # return gates1, gates2, combine, dispatch_mask

def torch_gating1(logits1, logits2, capacity=1):
    gates = F.softmax(logits1, dim=1)
    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    
    # Replace top-expert with min value
    logits_except1 = logits2.masked_fill(mask1.bool(), torch.finfo(logits1.dtype).min)
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)
    
    # return mask1, mask2, gates

    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.type_as(logits1), dim=0)
    res = me * ce * num_experts * num_experts
    
    # mask1_return, mask2_return = mask1.clone(), mask2.clone()
    # locations1_return, locations2_return = locations1.clone(), locations2.clone()
    
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    
    locations1 = locations1 * mask1
    locations2 = locations2 * mask2
    
    locations1_re, locations2_re = locations1.clone(), locations2.clone()
    mask1_re, mask2_re = mask1.clone(), mask2.clone()
    
    print(f"gates = {gates}")
    print(f"mask1 = {mask1}")
    print(f"mask2 = {mask2}")
    
    locations1_s = torch.sum(locations1, dim=1)
    locations2_s = torch.sum(locations2, dim=1)
    
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits1)
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity).type_as(logits1)
    
    # return locations1, locations2,  locations1_sc, locations2_sc
    
    mask1_float = mask1.type_as(logits1)
    mask2_float = mask2.type_as(logits1)
    
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)
    gates2_s = torch.einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
    gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
    
    print(f"gates1_s = ", gates1_s)
    print(f"mask1 = ", mask1_float)
    print(f"gates2_s = ", gates2_s)
    print(f"mask2 = ", mask2_float)
    
    # return gates, locations1_re, locations2_re, mask1_re, mask2_re, gates1, gates2
    
    # print(f"locations1_s: {locations1_s}")
    # print(f"locations2_s: {locations2_s}")
    # print(f"gates1: {gates1}")
    # print(f"gates2: {gates2}")
    
    combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    
    return gates, locations1_re, locations2_re, mask1_re, mask2_re, gates1, gates2, combine_weights, dispatch_mask
    # return gates, mask1, mask2, locations1_sc, locations2_sc, combine_weights, dispatch_mask
    # return locations1, locations2,  locations1_sc, locations2_sc, combine_weights

@triton.jit
def _2D_kernel(
    gates1_ptr,
    gates2_ptr,
    loca1_ptr,
    loca2_ptr,
    combine_ptr,
    stride_gates_row, stride_gates_col,
    stride_loca_row, stride_loca_col,
    stride_combine_s, stride_combine_e, stride_combine_c,
    s: tl.constexpr, e: tl.constexpr, c: tl.constexpr,
    BLOCK_SIZE_c: tl.constexpr,
):
    s_pid = tl.program_id(axis=0)
    e_pid = tl.program_id(axis=1)
    c_offset = tl.arange(0, BLOCK_SIZE_c)
    
    gates1_ptrs = gates1_ptr + s_pid * stride_gates_row + e_pid
    gates2_ptrs = gates2_ptr + s_pid * stride_gates_row + e_pid
    loca1_ptrs = loca1_ptr + s_pid * stride_loca_row + c_offset
    loca2_ptrs = loca2_ptr + s_pid * stride_loca_row + c_offset
    
    gates1_data = tl.load(gates1_ptrs)
    gates2_data = tl.load(gates2_ptrs)
    loca1_data = tl.load(loca1_ptrs, mask=c_offset < c)
    loca2_data = tl.load(loca2_ptrs, mask=c_offset < c)
    
    combine1 = gates1_data * loca1_data
    combine2 = gates2_data * loca2_data
    combine = combine1 + combine2
    
    combine_ptrs = combine_ptr + s_pid * stride_combine_s + e_pid * stride_combine_e + c_offset * stride_combine_c
    
    tl.store(combine_ptrs, combine, mask=c_offset < c)
    

def einsum2D(gates1, gates2, locations1, locations2):
    s, e = gates1.shape
    s, c = locations1.shape
    combine = torch.zeros((s, e, c), device=gates1.device)
    
    stride_gates_row, stride_gates_col = gates1.stride()
    stride_loca_row, stride_loca_col = locations1.stride()
    stride_combine_s, stride_combine_e, stride_combine_c = combine.stride()
    
    BLOCK_SIZE_c = triton.next_power_of_2(c)
    
    _2D_kernel[(s, e,)](
        gates1,
        gates2,
        locations1,
        locations2,
        combine,
        stride_gates_row, stride_gates_col,
        stride_loca_row, stride_loca_col,
        stride_combine_s, stride_combine_e, stride_combine_c,
        s=s, e=e, c=c,
        BLOCK_SIZE_c=BLOCK_SIZE_c,
    )
    
    return combine

def bench_fused():
    device = torch.device("cuda:0")
    logits1 = torch.randn(4096*16, 12).to(device)
    logits2 = torch.randn(4096*16, 12).to(device)
    
    # mask1_triton, mask2_triton, gates_triton = fused_gating1(logits1, logits2)
    # mask1_torch, mask2_torch, gates_torch = torch_gating1(logits1, logits2)
    
    # assert mask1_triton.shape == mask1_torch.shape
    # assert mask2_triton.shape == mask2_torch.shape
    # assert gates_triton.shape == gates_torch.shape
    # assert torch.allclose(mask1_triton, mask1_torch)
    # assert torch.allclose(mask2_triton, mask2_torch)
    # assert torch.allclose(gates_triton, gates_torch)
    
    # gates, mask1_torch, mask2_torch, local1_torch, local2_torch, res_torch, mask1_torch_, mask2_torch_ = torch_gating1(logits1, logits2)
    # local1_triton, local2_triton, res_triton, mask1_triton, mask2_triton = fused_gating2(mask1_torch, mask2_torch, gates)
    
    
    # assert local1_triton.shape == local1_torch.shape
    # assert local2_triton.shape == local2_torch.shape
    # assert res_torch.shape == res_triton.shape
    # assert mask1_triton.shape == mask1_torch_.shape
    # assert mask2_triton.shape == mask2_torch_.shape
    # assert torch.allclose(local1_triton, local1_torch)
    # assert torch.allclose(local2_triton, local2_torch)
    # assert torch.allclose(res_torch, res_triton)
    # assert torch.allclose(mask1_triton, mask1_torch_)
    # assert torch.allclose(mask2_triton, mask2_torch_)
    
    # loca1, loca2, locations1_torch, locations2_torch= torch_gating1(logits1, logits2, capacity=2048)
    # locations1_triton, locations2_triton = fused_gating3(loca1, loca2, capacity=2048)
    
    
    # assert locations1_triton.shape == locations1_torch.shape
    # assert locations2_triton.shape == locations2_torch.shape
    # assert torch.allclose(locations1_triton, locations1_torch)
    # assert torch.allclose(locations2_triton, locations2_torch)

    # gates, mask1, mask2, gates1_s_torch, gates2_s_torch= torch_gating1(logits1, logits2, capacity=2048)
    # gates1_s_triton, gates2_s_triton = fused_gating4(gates, mask1, mask2)
    
    
    # assert gates1_s_triton.shape == gates1_s_torch.shape
    # assert gates2_s_triton.shape == gates2_s_torch.shape
    # assert torch.allclose(gates1_s_triton, gates1_s_torch)
    # assert torch.allclose(gates2_s_triton, gates2_s_torch)
    
    # gates, mask1, mask2, locations1_sc, locations2_sc, combine_weights, dispatch_mask= torch_gating1(logits1, logits2, capacity=2048)
    # combine_weights_triton, mask_triton = fused_gating5(gates, mask1, mask2, locations1_sc, locations2_sc)
    # assert combine_weights_triton.shape == combine_weights.shape
    # assert torch.allclose(combine_weights_triton, combine_weights)
    # assert mask_triton.shape == dispatch_mask.shape
    # assert torch.allclose(mask_triton, dispatch_mask)
    
    gates, locations1_re, locations2_re, mask1_re, mask2_re, gates1, gates2,combine_weights, dispatch_mask= torch_gating1(logits1, logits2, capacity=2048)
    gates1_, gates2_, combine_weights_triton, mask_triton = fused_gating35(gates, locations1_re, locations2_re, mask1_re, mask2_re, c=2048)
    assert torch.allclose(gates1_, gates1)
    assert torch.allclose(gates2_, gates2)
    # import pdb; pdb.set_trace()
    
    assert combine_weights_triton.shape == combine_weights.shape
    assert torch.allclose(combine_weights_triton, combine_weights)
    assert mask_triton.shape == dispatch_mask.shape
    assert torch.allclose(mask_triton, dispatch_mask)


if __name__ == '__main__':
    # device = torch.device('cuda:0')
    # input = torch.arange(0,10).reshape(2, 5).to(device=device)
    
    # output_triton = sum_forward(input, dim=0, keepdim=True)
    # output_torch = torch.sum(input, dim=0, keepdim=True)
    
    # print(f"input dtype = {input.dtype}, output_triton dtype = {output_triton.dtype}, output_torch dtype = {output_torch.dtype}")
    
    # print(f"input = {input}")
    
    # print(f"output torch = {output_torch}")
    
    # if torch.allclose(output_torch, output_triton):
    #     print("triton implementation is equal to the torch implementation.")
    # else:
    #     print("triton implementation is not equal to the torch implementation.")
    
    # input1 = torch.arange(0, 16).to(device)
    # input2 = input1 * 10
    
    # output = add(input1, input2)
    
    # print(f"input1 = {input1}")
    # print(f"input2 = {input2}")
    # print(f"output shape = {output.shape}, output = {output}")
    
    # sum_dim0()
    # sum_dim1()
    
    # benchmark_sum()
    # benchmark_softmax()
    # benchmak_fused()
    # argmax()
    # benchmark_softmax_dim1()
    
    # one_hot()
    bench_fused()
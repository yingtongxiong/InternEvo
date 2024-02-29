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
    benchmak_fused()
    # argmax()
    # benchmark_softmax_dim1()
    
    # one_hot()
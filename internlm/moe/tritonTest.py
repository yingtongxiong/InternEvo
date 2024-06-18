import triton
import triton.language as tl
import torch

@triton.jit
def _softmax(Y, stride_ym, stride_yn, X, stride_xm, stride_xn, M, N):

    # row index
    m = tl.program_id(0)
    
    print("m = ", m,  flush=True)
    
    # BLOCK_SIZE = 1024
    # column index in block
    n = tl.arange(0, 4)
    
    X = X + m * stride_xm + n * stride_xn
    x = tl.load(X, mask=n < N, other=-float('inf'))
    
    # compute norm
    z = x - tl.max(x, axis=0)
    num = tl.exp(z)
    denom = tl.sum(num, axis=0)
    y = num / denom
    
    # write back to Y
    Y = Y + m * stride_ym + n * stride_yn
    tl.store(Y, y, mask=n < N)
    

def softmax(X, Y):
    # triton launch grid
    grid = (X.shape[0], )
    _softmax[grid](Y, Y.stride(0), Y.stride(1),
                   X, X.stride(0), X.stride(1),
                   X.shape[0], X.shape[1])

@triton.jit
def matmul_kernel(
    # pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # matrix dimension
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # meta parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # ACTIVATION: tl.constexpr,
):
    '''
    Kernel for computing the matmul C = A x B.
    A: (M, K), B: (K, N), C:(M, N)
    '''
    
    # pid: C中的block index, 以group为单位竖着数
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # 当前计算的block所在的group
    group_id = pid // num_pid_in_group
    # 每一个group的第一个block的全局block id, 在M维度
    first_pid_m = group_id * GROUP_SIZE_M
    # 最后一个group可能不足GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # 计算pid对应的block在C中的位置
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 通过累加A和B的block乘积来计算C的block输出
    # 首先找到A和B对应block的偏移量
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # a_ptrs, b_ptrs分别指A和B当前block的起始指针位置
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 遍历A和B的block并叠加到C的block
    # 计算C的一个block需要将A的M个block与B的N个block给叠加起来，上面的ptrs就指向了A和B的第一个block位置
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载A和B的数据
        # 有时候K并不能被BLOCK_SIZE_K整除，因此，对于每一行或者每一列的最后一个block都需要做一个判断
        # 对于实际大小不足BLOCK_SIZE_K的需要额外进行考虑，直接赋值0.0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 做实际运算
        accumulator += tl.dot(a, b)
        
        # 前移ptrs指针，进行下一个block计算
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    # 将累加的计算结果写回到C中
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    # check constraints
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous()
    assert b.is_contiguous()
    
    M, K = a.shape
    K, N = b.shape
    
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 1D launch kernel, where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # ACTIVATION=activation
        32, 32, 32, 2,
    )
    return c
    
@triton.jit
def bmatmul_kernel(
    # pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # matrix dimension
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    # meta parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # ACTIVATION: tl.constexpr,
):
    '''
    Kernel for computing the matmul C = A x B.
    A: (B, M, K), B: (B, K, N), C:(B, M, N)
    '''
    
    # pid: C中的block index, 以group为单位竖着数
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # 当前计算的block所在的group
    group_id = pid // num_pid_in_group
    # 每一个group的第一个block的全局block id, 在M维度
    first_pid_m = group_id * GROUP_SIZE_M
    # 最后一个group可能不足GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # 计算pid对应的block在C中的位置
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 通过累加A和B的block乘积来计算C的block输出
    # 首先找到A和B对应block的偏移量
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # a_ptrs, b_ptrs分别指A和B当前block的起始指针位置
    a_ptrs = a_ptr + (offs_b * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_b * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 遍历A和B的block并叠加到C的block
    # 计算C的一个block需要将A的M个block与B的N个block给叠加起来，上面的ptrs就指向了A和B的第一个block位置
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载A和B的数据
        # 有时候K并不能被BLOCK_SIZE_K整除，因此，对于每一行或者每一列的最后一个block都需要做一个判断
        # 对于实际大小不足BLOCK_SIZE_K的需要额外进行考虑，直接赋值0.0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 做实际运算
        accumulator += tl.dot(a, b)
        
        # 前移ptrs指针，进行下一个block计算
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    # 将累加的计算结果写回到C中
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_b * stride_cb + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def bmatmul(a, b):
    # check constraints
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    assert a.is_contiguous()
    assert b.is_contiguous()
    
    B, M, K = a.shape
    B, K, N = b.shape
    
    # allocates output
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    
    # 1D launch kernel, where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B,
    )
    bmatmul_kernel[grid](
        a, b, c,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        # ACTIVATION=activation
        32, 32, 32, 2,
    )
    return c

def unitest_bmatmul():
    torch.manual_seed(0)
    a = torch.randn((4, 512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((4, 512, 512), device='cuda', dtype=torch.float16)
    triton_output = bmatmul(a, b)
    torch_output = torch.bmm(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("Trinton and Torch match")
    else:
        print("Trinton and Torch differ")

def unitest_matmul():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("Trinton and Torch match")
    else:
        print("Trinton and Torch differ")

@triton.jit
def sum_kernel(
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
    
    a = tl.load(a_ptrs)
    a = tl.sum(a, axis=1)
    
    c_ptrs = c_ptr + pid * BLOCK_SIZE + v
    tl.store(c_ptrs, a)

def sum():
    device = torch.device('cuda', 0)
    a = torch.ones(128, 64, device=device) / 128
    c = torch.zeros(128, device=device)
    
    stride_a0, stride_a1 = a.stride()
    M, N = a.shape
    
    grid = lambda meta: (4,)
    sum_kernel[grid](
        a, c,
        stride_a0, stride_a1,
        M, N,
        BLOCK_SIZE=32,
    )
    
    expect_res = torch.sum(a, axis=1)
    
    print(f"sum kernel output = {c}")
    print(f"expected: {expect_res.shape}, {expect_res}")

if __name__ == '__main__':
    # X = torch.normal(0, 1, size=(2, 4), device='cuda')
    # Y = torch.empty_like(X)
    # softmax(X, Y)
    # print(Y)
    # unitest_matmul()
    # unitest_bmatmul()
    sum()
    
    
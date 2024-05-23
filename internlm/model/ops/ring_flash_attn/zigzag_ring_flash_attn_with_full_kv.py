import torch
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.parallel.comm.utils import all_gather_raw, reduce_scatter_raw

from .utils import RingComm, update_out_and_lse


def zigzag_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):
    assert causal is True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    # next_k, next_v = None, None
    # full_k = [k0, k7, k1, k6, k2, k5, k3, k4]
    # full_k, handle_full_k = all_gather_raw(k, process_group, async_op=True, gather_dim=1)
    # full_v, handle_full_v = all_gather_raw(v, process_group, async_op=True, gather_dim=1)

    def forward(q, k, v, causal):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    def get_next_kv_idx(prev_idx) -> int:
        if prev_idx == 0:
            return comm.world_size - 1
        else:
            return prev_idx - 1

    def _head_forward(q, full_k, full_v):
        out = None
        lse = None

        for step in range(comm.world_size):

            # # import pdb;pdb.set_trace()
            # if step + 1 != comm.world_size:
            #     next_k: torch.Tensor = comm.send_recv(k)
            #     next_v: torch.Tensor = comm.send_recv(v)
            #     comm.commit()

            # if step == 1:
            #     handle_full_v.wait()
            #     handle_full_k.wait()
            # handle_full_v.wait()
            # handle_full_k.wait()

            # full_k = [k0, k7, k1, k6, k2, k5, k3, k4]

            if step == 0:
                cur_kv_idx = comm.rank
                # assert k.equal(full_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len])
                # assert v.equal(full_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len])
                block_out, block_lse = forward(q, 
                                               full_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len], 
                                               full_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len], 
                                               causal=True)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            elif step <= comm.rank:
                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k0 = full_k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                v0 = full_v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k = full_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                v = full_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                q1 = q[:, block_seq_len:]
                block_out, block_lse = forward(q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )    
        # if step + 1 != comm.world_size:
        #     comm.wait()
        #     k = next_k
        #     v = next_v
        return out, lse
    
    head_chunks = 1
    head_step = k.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)
    
    outs = []
    lses = []
    
    for i in range(head_chunks):
        if i == 0:
            k_cur, handle_k_cur = all_gather_raw(k_splits[i], process_group, async_op=True, gather_dim=1)
            v_cur, handle_v_cur = all_gather_raw(v_splits[i], process_group, async_op=True, gather_dim=1)
            handle_k_cur.wait()
            handle_v_cur.wait()
        else:
            handle_k_next.wait()
            handle_v_next.wait()
            k_cur = k_next
            v_cur = v_next
    
        out, lse = _head_forward(q[..., i * head_step : (i + 1) * head_step, :], k_cur, v_cur)
        
        if i != head_chunks - 1:
            k_next, handle_k_next = all_gather_raw(k_splits[i + 1], process_group, async_op=True, gather_dim=1)
            v_next, handle_v_next = all_gather_raw(v_splits[i + 1], process_group, async_op=True, gather_dim=1)
        
        outs.append(out)
        lses.append(lse)
    
    out = torch.cat(outs, dim=-2).to(q.dtype)
    lse = torch.cat(lses, dim=-2).squeeze(dim=-1).transpose(1, 2)
    return out, lse


def zigzag_ring_flash_attn_backward_full_kv(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):
    if gpc.get_global_rank() == 0:
        print("full_kv_zigzag_with_full_dkv = False", flush=True)
    assert causal is True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    # next_k, next_v = None, None
    # dk_comm_buffer, dv_comm_buffer = None, None
    
    head_chunks = 2
    head_step = k.shape[-2] // head_chunks
    k_splits = torch.chunk(k, chunks=head_chunks, dim=-2)
    v_splits = torch.chunk(v, chunks=head_chunks, dim=-2)

    # full_k = [k0, k7, k1, k6, k2, k5, k3, k4]
    # full_k, handle_full_k = all_gather_raw(k, process_group, async_op=True, gather_dim=1)
    # full_v, handle_full_v = all_gather_raw(v, process_group, async_op=True, gather_dim=1)

    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def get_next_kv_idx(prev_idx) -> int:
        if prev_idx == 0:
            return kv_comm.world_size - 1
        else:
            return prev_idx - 1

    def backward(dout, q, k, v, out, softmax_lse, causal, chunk_idx):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q, chunk_idx * head_step : (chunk_idx + 1) * head_step, :],
            dk_buffer[:, :seqlen_kv, chunk_idx * head_step : (chunk_idx + 1) * head_step, :],
            dv_buffer[:, :seqlen_kv, chunk_idx * head_step : (chunk_idx + 1) * head_step, :],
            dropout_p,
            softmax_scale,
            causal,
        )


    def _head_backward(dout, q, k, v, out, softmax_lse, chunk_idx):
        
        # dq, dk, dv = None, None, None
        # next_dk, next_dv = None, None
        
        for step in range(kv_comm.world_size):
            # if step + 1 != kv_comm.world_size:
            #     next_k = kv_comm.send_recv(k)
            #     next_v = kv_comm.send_recv(v)
            #     kv_comm.commit()

            # if step == 1:
            #     handle_full_v.wait()
            #     handle_full_k.wait()
            
            # import pdb; pdb.set_trace()
            dk_comm_buffer, dv_comm_buffer = None, None
            
            if step == 0:
                
                cur_kv_idx = kv_comm.rank
                backward(dout, 
                         q, 
                         k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len], 
                         v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len], 
                         out, softmax_lse, causal=True, chunk_idx=chunk_idx)
                dq = dq_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :].to(torch.float32)
                dk = dk_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :].to(torch.float32)
                dv = dv_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :].to(torch.float32)
            else:
                if step <= kv_comm.rank:
                    k0 = k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                    v0 = v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                    backward(dout, q, k0, v0, out, softmax_lse, causal=False, chunk_idx=chunk_idx)
                    dq += dq_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :]
                else:
                    cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                    # print(f"xyt global rank = {gpc.get_global_rank()}, step = {step}, before k = {k.shape}, before v = {v.shape}", flush=True)
                    k1 = k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                    v1 = v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                    
                    dout1 = dout.chunk(2, dim=1)[1]
                    q1 = q.chunk(2, dim=1)[1]
                    out1 = out.chunk(2, dim=1)[1]
                    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
                    backward(dout1, q1, k1, v1, out1, softmax_lse1, causal=False, chunk_idx=chunk_idx)
                        
                    # always use the first half in dq_buffer.
                    dq[:, block_seq_len:] += dq_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :][:, :block_seq_len]

                d_kv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv

                if step <= kv_comm.rank:
                    dk[:, :block_seq_len] += dk_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :][:, :block_seq_len]
                    dv[:, :block_seq_len] += dv_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :][:, :block_seq_len]
                else:
                    dk += dk_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :]
                    dv += dv_buffer[..., chunk_idx * head_step : (chunk_idx + 1) * head_step, :]

            # if step + 1 != kv_comm.world_size:
            #     kv_comm.wait()
            #     k = next_k
            #     v = next_v

            next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
            next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
            d_kv_comm.commit()
        
        return dq, next_dk, next_dv
    
    dqs, next_dks, next_dvs = [], [], []
    
    for i in range(head_chunks):
        if i == 0:
            # all gather the first part of k_splits ,v_splits
            k_cur, handle_k_cur = all_gather_raw(k_splits[i], process_group, async_op=True, gather_dim=1)
            v_cur, handle_v_cur = all_gather_raw(v_splits[i], process_group, async_op=True, gather_dim=1)
            handle_k_cur.wait()
            handle_v_cur.wait()
        else:
            handle_k_next.wait()
            handle_v_next.wait()
            k_cur = k_next
            v_cur = v_next
        
        # all gather the next part of k_splits, v_splits
        if i != head_chunks - 1:
            k_next, handle_k_next = all_gather_raw(k_splits[i + 1], process_group, async_op=True, gather_dim=1)
            v_next, handle_v_next = all_gather_raw(v_splits[i + 1], process_group, async_op=True, gather_dim=1)

        dq, next_dk, next_dv = _head_backward(dout[..., i * head_step : (i + 1) * head_step, :], 
                                              q[..., i * head_step : (i + 1) * head_step, :], 
                                              k_cur, 
                                              v_cur,
                                              out[..., i * head_step : (i + 1) * head_step, :], 
                                              softmax_lse[..., i * head_step : (i + 1) * head_step, :], 
                                              chunk_idx=i)
        d_kv_comm.wait()
        dqs.append(dq)
        next_dks.append(next_dk)
        next_dvs.append(next_dv)
    
    dq = torch.cat(dqs, dim=-2).to(q.dtype)
    next_dk = torch.cat(next_dks, dim=-2).to(q.dtype)
    next_dv = torch.cat(next_dvs, dim=-2).to(q.dtype)

    return dq, next_dk, next_dv


def zigzag_ring_flash_attn_backward_full_kv_dkv(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):
    if gpc.get_global_rank() == 0:
        print("full_kv_zigzag_with_full_dkv = True", flush=True)
    assert causal is True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    # d_kv_comm = RingComm(process_group)

    dq, dk, dv = None, None, None
    # next_dk, next_dv = None, None
    # next_k, next_v = None, None
    # dk_comm_buffer, dv_comm_buffer = None, None

    # full_k = [k0, k7, k1, k6, k2, k5, k3, k4]
    full_dk, full_dv = None, None
    full_k, handle_full_k = all_gather_raw(k, process_group, async_op=True, gather_dim=1)
    full_v, handle_full_v = all_gather_raw(v, process_group, async_op=True, gather_dim=1)

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def get_next_kv_idx(prev_idx) -> int:
        if prev_idx == 0:
            return kv_comm.world_size - 1
        else:
            return prev_idx - 1

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal,
        )

    for step in range(kv_comm.world_size):
        # if step + 1 != kv_comm.world_size:
        #     next_k = kv_comm.send_recv(k)
        #     next_v = kv_comm.send_recv(v)
        #     kv_comm.commit()

        if step == 1:
            handle_full_v.wait()
            handle_full_k.wait()
            full_dk = torch.empty_like(full_k, dtype=torch.float32)
            full_dv = torch.empty_like(full_v, dtype=torch.float32)
            full_dk[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len].copy_(dk)
            full_dv[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len].copy_(dv)

        if step == 0:
            cur_kv_idx = kv_comm.rank
            backward(dout, q, k, v, out, softmax_lse, causal=True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k0 = full_k[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                v0 = full_v[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer
            else:
                cur_kv_idx = get_next_kv_idx(cur_kv_idx)
                k = full_k[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                v = full_v[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len]
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            # d_kv_comm.wait()
            # dk_comm_buffer, dv_comm_buffer = dk, dv
            # dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                full_dk[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len].copy_(
                    dk_buffer[:, :block_seq_len].to(torch.float32)
                )
                full_dv[:, 2 * cur_kv_idx * block_seq_len : (2 * cur_kv_idx + 1) * block_seq_len].copy_(
                    dv_buffer[:, :block_seq_len].to(torch.float32)
                )
            else:
                full_dk[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len].copy_(
                    dk_buffer.to(torch.float32)
                )
                full_dv[:, 2 * cur_kv_idx * block_seq_len : 2 * (cur_kv_idx + 1) * block_seq_len].copy_(
                    dv_buffer.to(torch.float32)
                )

        # if step + 1 != kv_comm.world_size:
        #     kv_comm.wait()
        #     k = next_k
        #     v = next_v

        # next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        # next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        # d_kv_comm.commit()

    # d_kv_comm.wait()

    # reduce-scatter dk and kv
    dk, _ = reduce_scatter_raw(full_dk, process_group, async_op=False, reduce_dim=1)
    dv, _ = reduce_scatter_raw(full_dv, process_group, async_op=False, reduce_dim=1)

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    """ZigZagRingFlashAttnFunc"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        with_full_dkv,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()

        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group, q, k, v, softmax_scale=softmax_scale, dropout_p=dropout_p, causal=causal
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.with_full_dkv = with_full_dkv
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):  # pylint: disable=W0613
        q, k, v, out, softmax_lse = ctx.saved_tensors
        with_full_dkv = ctx.with_full_dkv

        backward_func = (
            zigzag_ring_flash_attn_backward_full_kv_dkv if with_full_dkv else zigzag_ring_flash_attn_backward_full_kv
        )

        dq, dk, dv = backward_func(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_qkvpacked_func_with_full_kv(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        gpc.config.get("full_kv_zigzag_with_full_dkv", False),  # TODO: pass by args.
        group,
    )


def zigzag_ring_flash_attn_kvpacked_func_with_full_kv(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        gpc.config.get("full_kv_zigzag_with_full_dkv", False),  # TODO: pass by args.
        group,
    )


def zigzag_ring_flash_attn_func_with_full_kv(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        gpc.config.get("full_kv_zigzag_with_full_dkv", False),  # TODO: pass by args.
        group,
    )

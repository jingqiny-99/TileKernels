# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton fused kernels for mHC (Manifold-Constrained Hyper-Connections).

Drop-in replacements for the cuTile kernels in ``fused_mhc_kernels.py``.
Enabled via ``mhc_use_triton_kernels=True`` in TransformerConfig.

Five fused operations (matching the cuTile API):
  - sinkhorn:          Sinkhorn-Knopp projection to doubly stochastic matrix
  - h_aggregate:       weighted n-stream -> 1-stream aggregation
  - h_post_bda:        fused H_res @ residual + H_post * (x + bias)
  - proj_rms:          fused projection + RMS normalization
  - proj_rms_compute_h: fused projection + RMS + compute_h activations
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

LOG2E = tl.constexpr(1.4426950408889634)


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return max(n, 1)


# ============================================================================
# 1. Sinkhorn-Knopp
# ============================================================================


@triton.autotune(
    configs=[triton.Config({}, num_warps=nw) for nw in (1, 2, 4, 8)],
    key=["HC", "NUM_ITERS"],
)
@triton.jit
def _triton_sinkhorn_fwd_kernel(
    inp_ptr, out_ptr, M_init_ptr,
    N_batch, eps,
    HC: tl.constexpr, NUM_ITERS: tl.constexpr,
):
    """Grid: (N_batch,). Each program handles one [HC, HC] matrix."""
    pid = tl.program_id(0)
    if pid >= N_batch:
        return
    base = pid * HC * HC
    offs_r = tl.arange(0, HC)
    offs_c = tl.arange(0, HC)
    # Load [HC, HC] logits
    ptrs = inp_ptr + base + offs_r[:, None] * HC + offs_c[None, :]
    logits = tl.load(ptrs).to(tl.float32)
    # exp2(logits - row_max) * LOG2E
    row_max = tl.max(logits, axis=1)  # [HC]
    M = tl.exp2((logits - row_max[:, None]) * LOG2E)
    # Save M_init
    mat_ptrs = base + offs_r[:, None] * HC + offs_c[None, :]
    tl.store(M_init_ptr + mat_ptrs, M.to(M_init_ptr.dtype.element_ty))
    # Iterative row/col normalization
    for _ in range(NUM_ITERS):
        row_sum = tl.sum(M, axis=1)  # [HC]
        M = M / (row_sum[:, None] + eps)
        col_sum = tl.sum(M, axis=0)  # [HC]
        M = M / (col_sum[None, :] + eps)
    tl.store(out_ptr + mat_ptrs, M.to(out_ptr.dtype.element_ty))


@triton.autotune(
    configs=[triton.Config({}, num_warps=nw) for nw in (1, 2, 4, 8)],
    key=["HC", "NUM_ITERS"],
)
@triton.jit
def _triton_sinkhorn_bwd_kernel(
    grad_out_ptr, M_init_ptr, grad_inp_ptr,
    ws_M_ptr, ws_rs_ptr, ws_cs_ptr,
    N_batch, eps,
    HC: tl.constexpr, NUM_ITERS: tl.constexpr,
):
    """Grid: (N_batch,). Each program handles one [HC, HC] backward."""
    pid = tl.program_id(0)
    if pid >= N_batch:
        return
    base = pid * HC * HC
    M_ws_base = pid * 2 * NUM_ITERS * HC * HC
    v_ws_base = pid * NUM_ITERS
    offs_r = tl.arange(0, HC)
    offs_c = tl.arange(0, HC)

    # Forward pass: recompute and save intermediates
    # Helper ptrs for [HC, HC] matrix at base offset
    mat_ptrs = base + offs_r[:, None] * HC + offs_c[None, :]

    M = tl.load(M_init_ptr + mat_ptrs).to(tl.float32)
    for t in range(NUM_ITERS):
        # Save M before row norm
        ws_off = M_ws_base + (2 * t) * HC * HC
        tl.store(ws_M_ptr + ws_off + offs_r[:, None] * HC + offs_c[None, :], M)
        # row_sum: [HC] (1D), then broadcast to [HC, 1] via [:, None]
        row_sum_1d = tl.sum(M, axis=1)  # [HC]
        tl.store(ws_rs_ptr + (v_ws_base + t) * HC + offs_r, row_sum_1d)
        M = M / (row_sum_1d[:, None] + eps)
        # Save M after row norm
        ws_off = M_ws_base + (2 * t + 1) * HC * HC
        tl.store(ws_M_ptr + ws_off + offs_r[:, None] * HC + offs_c[None, :], M)
        # col_sum: [HC] (1D), then broadcast to [1, HC] via [None, :]
        col_sum_1d = tl.sum(M, axis=0)  # [HC]
        tl.store(ws_cs_ptr + (v_ws_base + t) * HC + offs_c, col_sum_1d)
        M = M / (col_sum_1d[None, :] + eps)

    # Backward pass.  M tracks the forward output at the current reverse step:
    # after col norm for the col-normalization VJP, then after row norm for
    # the row-normalization VJP.
    grad = tl.load(grad_out_ptr + mat_ptrs).to(tl.float32)
    for t_rev in range(NUM_ITERS):
        t = NUM_ITERS - 1 - t_rev
        # Undo col normalization
        col_s = tl.load(ws_cs_ptr + (v_ws_base + t) * HC + offs_c).to(tl.float32)
        grad = grad / (col_s[None, :] + eps)
        col_corr = tl.sum(grad * M, axis=0)  # [HC]
        grad = grad - col_corr[None, :]
        M = tl.load(
            ws_M_ptr + M_ws_base + (2 * t + 1) * HC * HC + offs_r[:, None] * HC + offs_c[None, :]
        ).to(tl.float32)
        # Undo row normalization
        row_s = tl.load(ws_rs_ptr + (v_ws_base + t) * HC + offs_r).to(tl.float32)
        grad = grad / (row_s[:, None] + eps)
        row_corr = tl.sum(grad * M, axis=1)  # [HC]
        grad = grad - row_corr[:, None]
        M = tl.load(
            ws_M_ptr + M_ws_base + (2 * t) * HC * HC + offs_r[:, None] * HC + offs_c[None, :]
        ).to(tl.float32)

    # grad *= M_init (chain rule for exp2)
    M_init = tl.load(M_init_ptr + mat_ptrs).to(tl.float32)
    grad = grad * M_init
    tl.store(grad_inp_ptr + mat_ptrs, grad.to(grad_inp_ptr.dtype.element_ty))


def _triton_sinkhorn_fwd(input_logits, num_iterations, eps=1e-8):
    original_shape = input_logits.shape
    hc = original_shape[-1]
    N_batch = input_logits.numel() // (hc * hc)
    dev = input_logits.device
    out = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    M_init = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    inp = input_logits.contiguous().view(N_batch, hc, hc)
    grid = (N_batch,)
    _triton_sinkhorn_fwd_kernel[grid](inp, out, M_init, N_batch, eps, hc, num_iterations)
    return out.view(original_shape), M_init.view(original_shape)


def _triton_sinkhorn_bwd(grad_output, M_init, num_iterations, eps=1e-8):
    original_shape = grad_output.shape
    hc = original_shape[-1]
    N_batch = grad_output.numel() // (hc * hc)
    dev = grad_output.device
    grad_input = torch.empty(N_batch, hc, hc, dtype=grad_output.dtype, device=dev)
    go = grad_output.contiguous().view(N_batch, hc, hc)
    mi = M_init.contiguous().view(N_batch, hc, hc)
    ws_M = torch.empty(N_batch * 2 * num_iterations * hc * hc, dtype=torch.float32, device=dev)
    ws_rs = torch.empty(N_batch * num_iterations * hc, dtype=torch.float32, device=dev)
    ws_cs = torch.empty(N_batch * num_iterations * hc, dtype=torch.float32, device=dev)
    grid = (N_batch,)
    _triton_sinkhorn_bwd_kernel[grid](
        go, mi, grad_input, ws_M, ws_rs, ws_cs, N_batch, eps, hc, num_iterations,
    )
    return grad_input.view(original_shape)


# ============================================================================
# 2. H_aggregate
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_h_agg_fwd_kernel(
    x_ptr, h_ptr, out_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_x_s, stride_x_n, stride_x_c,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """out[s, c] = sum_i x[s, i, c] * h[s, i].  Grid: (cdiv(sb, BS), cdiv(C, BC))."""
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < sb
    mask_c = offs_c < C
    mask_2d = mask_s[:, None] & mask_c[None, :]

    acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
    for i in tl.static_range(N):
        x_i = tl.load(
            x_ptr + offs_s[:, None] * stride_x_s + i * stride_x_n + offs_c[None, :],
            mask=mask_2d, other=0.0,
        ).to(tl.float32)
        h_i = tl.load(h_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(tl.float32)
        acc += h_i[:, None] * x_i
    tl.store(out_ptr + offs_s[:, None] * C + offs_c[None, :], acc.to(out_ptr.dtype.element_ty), mask=mask_2d)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_h_agg_bwd_kernel(
    go_ptr, x_ptr, h_ptr, gx_ptr, gh_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_x_s, stride_x_n, stride_x_c,
    stride_gx_s, stride_gx_n, stride_gx_c,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """gx[s,i,c] = go[s,c]*h[s,i], gh[s,i] = sum_c go[s,c]*x[s,i,c].  Grid: (cdiv(sb,BS),)."""
    pid_s = tl.program_id(0)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < sb

    gh_acc = tl.zeros((BLOCK_S, N), dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask_2d = mask_s[:, None] & mask_c[None, :]

        go_tile = tl.load(
            go_ptr + offs_s[:, None] * C + offs_c[None, :], mask=mask_2d, other=0.0,
        ).to(tl.float32)

        for i in tl.static_range(N):
            h_i = tl.load(h_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(tl.float32)
            # gx[s, i, c] = go[s, c] * h[s, i]
            gx_tile = go_tile * h_i[:, None]
            tl.store(
                gx_ptr + offs_s[:, None] * stride_gx_s + i * stride_gx_n + offs_c[None, :],
                gx_tile.to(gx_ptr.dtype.element_ty), mask=mask_2d,
            )
            # gh[s, i] += sum_c go[s, c] * x[s, i, c]
            x_i = tl.load(
                x_ptr + offs_s[:, None] * stride_x_s + i * stride_x_n + offs_c[None, :],
                mask=mask_2d, other=0.0,
            ).to(tl.float32)
            dot = tl.sum(go_tile * x_i, axis=1)
            gh_acc += tl.where(tl.arange(0, N)[None, :] == i, dot[:, None], tl.zeros((BLOCK_S, N), dtype=tl.float32))

    offs_n = tl.arange(0, N)
    tl.store(gh_ptr + offs_s[:, None] * N + offs_n[None, :], gh_acc.to(gh_ptr.dtype.element_ty), mask=mask_s[:, None])


def _triton_h_aggregate_fwd(x, h_pre):
    s, b, n, C = x.shape
    sb = s * b
    out = torch.empty(sb, C, dtype=x.dtype, device=x.device)
    x_flat = x.contiguous().view(sb, n, C)
    h_flat = h_pre.contiguous().view(sb, n)
    grid = lambda META: (triton.cdiv(sb, META["BLOCK_S"]), triton.cdiv(C, META["BLOCK_C"]))
    _triton_h_agg_fwd_kernel[grid](
        x_flat, h_flat, out, sb, C, n,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
    )
    return out.view(s, b, C)


def _triton_h_aggregate_bwd(grad_output, x, h_pre):
    s, b, n, C = x.shape
    sb = s * b
    gx = torch.empty(sb, n, C, dtype=x.dtype, device=x.device)
    gh = torch.empty(sb, n, dtype=h_pre.dtype, device=x.device)
    go_flat = grad_output.contiguous().view(sb, C)
    x_flat = x.contiguous().view(sb, n, C)
    h_flat = h_pre.contiguous().view(sb, n)
    grid = lambda META: (triton.cdiv(sb, META["BLOCK_S"]),)
    _triton_h_agg_bwd_kernel[grid](
        go_flat, x_flat, h_flat, gx, gh, sb, C, n,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        gx.stride(0), gx.stride(1), gx.stride(2),
    )
    return gx.view(s, b, n, C), gh.view(s, b, n)


# ============================================================================
# 3. H_post BDA
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_hpb_fwd_kernel(
    hr_ptr, orig_ptr, hp_ptr, x_ptr, bias_ptr, out_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_hr_s, stride_hr_i, stride_hr_j,
    stride_orig_s, stride_orig_n, stride_orig_c,
    stride_out_s, stride_out_n, stride_out_c,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """out = hr @ orig + hp * (x + bias).  Grid: (cdiv(sb, BS), cdiv(C, BC))."""
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < sb
    mask_c = offs_c < C
    mask_2d = mask_s[:, None] & mask_c[None, :]

    # Load x (+ bias)  [BLOCK_S, BLOCK_C]
    x_tile = tl.load(x_ptr + offs_s[:, None] * C + offs_c[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        x_tile += bias_tile[None, :]

    for i in tl.static_range(N):
        # hp[s, i] * (x + bias)
        hp_i = tl.load(hp_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(tl.float32)
        out_i = hp_i[:, None] * x_tile

        # + sum_j hr[s, i, j] * orig[s, j, :]
        for j in tl.static_range(N):
            hr_ij = tl.load(
                hr_ptr + offs_s * stride_hr_s + i * stride_hr_i + j * stride_hr_j,
                mask=mask_s, other=0.0,
            ).to(tl.float32)
            orig_j = tl.load(
                orig_ptr + offs_s[:, None] * stride_orig_s + j * stride_orig_n + offs_c[None, :],
                mask=mask_2d, other=0.0,
            ).to(tl.float32)
            out_i += hr_ij[:, None] * orig_j

        tl.store(
            out_ptr + offs_s[:, None] * stride_out_s + i * stride_out_n + offs_c[None, :],
            out_i.to(out_ptr.dtype.element_ty), mask=mask_2d,
        )


# --- h_post_bda backward kernels (existing) ---

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_hpb_bwd_g_x_orig_kernel(
    go_ptr, hr_ptr, hp_ptr, g_orig_ptr, g_x_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_go_s, stride_go_n, stride_go_c,
    stride_hr_s, stride_hr_i, stride_hr_j,
    stride_orig_s, stride_orig_n, stride_orig_c,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """g_x = hp @ go, g_orig = hr.T @ go.  Grid: (cdiv(sb,BLOCK_S), cdiv(C,BLOCK_C))."""
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < sb
    mask_c = offs_c < C
    mask_2d = mask_s[:, None] & mask_c[None, :]

    g_x_acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
    for j in tl.static_range(N):
        go_j = tl.load(
            go_ptr + offs_s[:, None] * stride_go_s + j * stride_go_n + offs_c[None, :],
            mask=mask_2d, other=0.0,
        ).to(tl.float32)
        hp_j = tl.load(hp_ptr + offs_s * N + j, mask=mask_s, other=0.0).to(tl.float32)
        g_x_acc += hp_j[:, None] * go_j
    tl.store(g_x_ptr + offs_s[:, None] * C + offs_c[None, :], g_x_acc.to(g_x_ptr.dtype.element_ty), mask=mask_2d)

    for i in tl.static_range(N):
        g_orig_i = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(N):
            go_j = tl.load(
                go_ptr + offs_s[:, None] * stride_go_s + j * stride_go_n + offs_c[None, :],
                mask=mask_2d, other=0.0,
            ).to(tl.float32)
            hr_ji = tl.load(
                hr_ptr + offs_s * stride_hr_s + j * stride_hr_i + i * stride_hr_j,
                mask=mask_s, other=0.0,
            ).to(tl.float32)
            g_orig_i += hr_ji[:, None] * go_j
        tl.store(
            g_orig_ptr + offs_s[:, None] * stride_orig_s + i * stride_orig_n + offs_c[None, :],
            g_orig_i.to(g_orig_ptr.dtype.element_ty), mask=mask_2d,
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_hpb_bwd_g_hp_hr_kernel(
    go_ptr, orig_ptr, x_ptr, bias_ptr,
    g_hr_ptr, g_hp_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_go_s, stride_go_n, stride_go_c,
    stride_orig_s, stride_orig_n, stride_orig_c,
    stride_hr_s, stride_hr_i, stride_hr_j,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """g_hp = sum_c go*(x+bias), g_hr = go @ orig.T.  Grid: (cdiv(sb,BLOCK_S),)."""
    pid_s = tl.program_id(0)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < sb

    g_hp_acc = tl.zeros((BLOCK_S, N), dtype=tl.float32)
    g_hr_acc = tl.zeros((BLOCK_S, N * N), dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask_2d = mask_s[:, None] & mask_c[None, :]

        x_tile = tl.load(x_ptr + offs_s[:, None] * C + offs_c[None, :], mask=mask_2d, other=0.0).to(tl.float32)
        if HAS_BIAS:
            bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            x_tile += bias_tile[None, :]

        for i in tl.static_range(N):
            go_i = tl.load(
                go_ptr + offs_s[:, None] * stride_go_s + i * stride_go_n + offs_c[None, :],
                mask=mask_2d, other=0.0,
            ).to(tl.float32)
            dot_hp = tl.sum(go_i * x_tile, axis=1)
            g_hp_acc += tl.where(
                tl.arange(0, N)[None, :] == i, dot_hp[:, None], tl.zeros((BLOCK_S, N), dtype=tl.float32),
            )
            for j in tl.static_range(N):
                orig_j = tl.load(
                    orig_ptr + offs_s[:, None] * stride_orig_s + j * stride_orig_n + offs_c[None, :],
                    mask=mask_2d, other=0.0,
                ).to(tl.float32)
                dot_hr = tl.sum(go_i * orig_j, axis=1)
                g_hr_acc += tl.where(
                    tl.arange(0, N * N)[None, :] == i * N + j, dot_hr[:, None],
                    tl.zeros((BLOCK_S, N * N), dtype=tl.float32),
                )

    offs_n = tl.arange(0, N)
    tl.store(g_hp_ptr + offs_s[:, None] * N + offs_n[None, :], g_hp_acc.to(g_hp_ptr.dtype.element_ty), mask=mask_s[:, None])

    nn_offs = tl.arange(0, N * N)
    for i in tl.static_range(N):
        for j in tl.static_range(N):
            col_mask = (nn_offs == (i * N + j)).to(tl.float32)
            val = tl.sum(g_hr_acc * col_mask[None, :], axis=1)
            tl.store(
                g_hr_ptr + offs_s * stride_hr_s + i * stride_hr_i + j * stride_hr_j,
                val.to(g_hr_ptr.dtype.element_ty), mask=mask_s,
            )


# --- h_post_bda wrappers ---

def _triton_h_post_bda_fwd(h_res, original_residual, h_post, x, bias):
    s, b, n, C = original_residual.shape
    sb = s * b
    dev = h_res.device
    out = torch.empty(sb, n, C, dtype=h_res.dtype, device=dev)
    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, C)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, C)

    grid = lambda META: (triton.cdiv(sb, META["BLOCK_S"]), triton.cdiv(C, META["BLOCK_C"]))
    _triton_hpb_fwd_kernel[grid](
        hr_flat, orig_flat, hp_flat, x_flat, bias if bias is not None else x_flat, out,
        sb, C, n,
        hr_flat.stride(0), hr_flat.stride(1), hr_flat.stride(2),
        orig_flat.stride(0), orig_flat.stride(1), orig_flat.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_BIAS=(bias is not None),
    )
    return out.view(s, b, n, C)


def _triton_h_post_bda_bwd(grad_output, h_res, original_residual, h_post, x, bias):
    s, b, n, C = original_residual.shape
    sb = s * b
    dev = h_res.device

    g_hr = torch.empty(sb, n, n, dtype=h_res.dtype, device=dev)
    g_res = torch.empty(sb, n, C, dtype=original_residual.dtype, device=dev)
    g_hp = torch.empty(sb, n, dtype=h_post.dtype, device=dev)
    g_x = torch.empty(sb, C, dtype=x.dtype, device=dev)

    go_flat = grad_output.contiguous().view(sb, n, C)
    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, C)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, C)

    grid_a = lambda META: (triton.cdiv(sb, META["BLOCK_S"]), triton.cdiv(C, META["BLOCK_C"]))
    _triton_hpb_bwd_g_x_orig_kernel[grid_a](
        go_flat, hr_flat, hp_flat, g_res, g_x, sb, C, n,
        go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
        hr_flat.stride(0), hr_flat.stride(1), hr_flat.stride(2),
        g_res.stride(0), g_res.stride(1), g_res.stride(2),
    )

    grid_b = lambda META: (triton.cdiv(sb, META["BLOCK_S"]),)
    _triton_hpb_bwd_g_hp_hr_kernel[grid_b](
        go_flat, orig_flat, x_flat, bias if bias is not None else x_flat,
        g_hr, g_hp, sb, C, n,
        go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
        orig_flat.stride(0), orig_flat.stride(1), orig_flat.stride(2),
        g_hr.stride(0), g_hr.stride(1), g_hr.stride(2),
        HAS_BIAS=(bias is not None),
    )

    g_bias = g_x.sum(dim=0).to(dtype=bias.dtype) if bias is not None else None
    return (
        g_hr.view(s, b, n, n),
        g_res.view(s, b, n, C),
        g_hp.view(s, b, n),
        g_x.view(s, b, C),
        g_bias,
    )


# ============================================================================
# 4-5. Fused proj_rms_compute_h  (matmul + norm + activations)
#
# Standalone proj_rms is not performance-critical — use PyTorch ops.
# All kernel work goes into the fused compute_h path.
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for bm in (32, 64, 128)
        for bk in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 4)
    ],
    key=["M", "N_PAD", "K"],
)
@triton.jit
def _triton_matmul_norm_kernel(
    A_ptr, B_ptr, PROJ_ptr, SUM_SQ_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """proj = A @ B^T, sum_sq = sum(A^2, axis=1).

    Grid: (cdiv(M, BLOCK_M),).  Outputs proj [M, N] and sum_sq [M].
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask_mk = mask_m[:, None] & mask_k[None, :]

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(a * a, axis=1)

        mask_nk = mask_n[:, None] & mask_k[None, :]
        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
            mask=mask_nk, other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, tl.trans(b))

    mask_mn = mask_m[:, None] & mask_n[None, :]
    tl.store(PROJ_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(PROJ_ptr.dtype.element_ty), mask=mask_mn)
    tl.store(SUM_SQ_ptr + offs_m, sum_sq.to(SUM_SQ_ptr.dtype.element_ty), mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for bm in (32, 64, 128)
        for bk in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 4)
    ],
    key=["M", "N_PAD", "K", "SPLIT_K"],
)
@triton.jit
def _triton_matmul_norm_split_k_kernel(
    A_ptr, B_ptr, PROJ_ACC_ptr, SUM_SQ_ACC_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    N_PAD: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Split-K proj = A @ B^T partials and sum_sq partials.

    Grid: (cdiv(M, BLOCK_M), SPLIT_K).  Partial outputs are laid out as
    [SPLIT_K, M, N] and [SPLIT_K, M].
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    blocks_per_split = (num_k_blocks + SPLIT_K - 1) // SPLIT_K
    first_block = pid_k * blocks_per_split
    last_block = tl.minimum(first_block + blocks_per_split, num_k_blocks)

    acc = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for block_offset in tl.static_range(0, blocks_per_split):
        k_block = first_block + block_offset
        k_start = k_block * BLOCK_K
        in_split = k_block < last_block
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask_mk = mask_m[:, None] & mask_k[None, :] & in_split

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(a * a, axis=1)

        mask_nk = mask_n[:, None] & mask_k[None, :] & in_split
        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
            mask=mask_nk, other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, tl.trans(b))

    split_proj_base = pid_k * M * N
    split_sum_base = pid_k * M
    mask_mn = mask_m[:, None] & mask_n[None, :]
    tl.store(
        PROJ_ACC_ptr + split_proj_base + offs_m[:, None] * N + offs_n[None, :],
        acc.to(PROJ_ACC_ptr.dtype.element_ty),
        mask=mask_mn,
    )
    tl.store(
        SUM_SQ_ACC_ptr + split_sum_base + offs_m,
        sum_sq.to(SUM_SQ_ACC_ptr.dtype.element_ty),
        mask=mask_m,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=nw)
        for bm in (32, 64, 128)
        for nw in (2, 4, 8)
    ],
    key=["M", "N", "SPLIT_K"],
)
@triton.jit
def _triton_compute_h_reduce_kernel(
    PROJ_ACC_ptr, SUM_SQ_ACC_ptr, Bias_ptr,
    Alpha_pre_ptr, Alpha_post_ptr, Alpha_res_ptr,
    Y_ptr, PROJ_OUT_ptr, R_ptr,
    M, N, K: tl.constexpr,
    n: tl.constexpr, eps,
    N_PAD: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Reduce split-K partials, compute r, and apply compute_h activations."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N
    mask_mn = mask_m[:, None] & mask_n[None, :]

    proj = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for split in tl.static_range(SPLIT_K):
        proj += tl.load(
            PROJ_ACC_ptr + split * M * N + offs_m[:, None] * N + offs_n[None, :],
            mask=mask_mn, other=0.0,
        ).to(tl.float32)
        sum_sq += tl.load(
            SUM_SQ_ACC_ptr + split * M + offs_m,
            mask=mask_m, other=0.0,
        ).to(tl.float32)

    tl.store(PROJ_OUT_ptr + offs_m[:, None] * N + offs_n[None, :], proj.to(PROJ_OUT_ptr.dtype.element_ty), mask=mask_mn)

    r = tl.sqrt(sum_sq / float(K))
    tl.store(R_ptr + offs_m, r.to(R_ptr.dtype.element_ty), mask=mask_m)

    alpha_pre = tl.load(Alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(Alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(Alpha_res_ptr).to(tl.float32)
    mask_pre_n = (offs_n < n).to(tl.float32)
    mask_post_n = ((offs_n >= n) & (offs_n < 2 * n)).to(tl.float32)
    mask_res_n = (offs_n >= 2 * n).to(tl.float32)
    scale = alpha_pre * mask_pre_n + alpha_post * mask_post_n + alpha_res * mask_res_n

    bias_val = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    linear = proj * scale[None, :] / (r[:, None] + eps) + bias_val[None, :]
    sig = tl.sigmoid(linear)
    out = sig * mask_pre_n[None, :] + 2.0 * sig * mask_post_n[None, :] + linear * mask_res_n[None, :]

    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], out.to(Y_ptr.dtype.element_ty), mask=mask_mn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for bm in (32, 64, 128)
        for bk in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 4)
    ],
    key=["M", "N_PAD", "K"],
)
@triton.jit
def _triton_grad_x_weight_kernel(
    A_ptr, B_ptr, R_ptr, DD_ptr, DR_ptr, DA_ptr, DB_ptr,
    M, N, K: tl.constexpr,
    eps,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """grad_x = grad_proj @ weight + rms_dnorm, grad_weight = grad_proj^T @ x.

    Grid: (cdiv(K, BLOCK_K),).  Loops over M.
    R_ptr stores r = norm/sqrt(K).  DR_ptr stores grad_r_total.
    """
    pid_k = tl.program_id(0)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    mask_nk = mask_n[:, None] & mask_k[None, :]
    b_tile = tl.load(
        B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
        mask=mask_nk, other=0.0,
    ).to(tl.float32)

    acc_db = tl.zeros((BLOCK_K, N_PAD), dtype=tl.float32)
    NUM_M_TILES = tl.cdiv(M, BLOCK_M)

    for m_tile in range(NUM_M_TILES):
        offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask_mk = mask_m[:, None] & mask_k[None, :]

        a_tile = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_mk, other=0.0,
        ).to(tl.float32)

        # r = norm/sqrt(K).  Simplified rms_dnorm: grad_r_total * x / (r * K)
        r_val = tl.load(R_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)
        dr_val = tl.load(DR_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        inv_rK = 1.0 / (r_val * float(K))
        rms_grad = (dr_val * inv_rK)[:, None] * a_tile

        mask_mn = mask_m[:, None] & mask_n[None, :]
        dd_tile = tl.load(
            DD_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=mask_mn, other=0.0,
        ).to(tl.float32)

        da_tile = tl.dot(dd_tile, b_tile) + rms_grad
        tl.store(DA_ptr + offs_m[:, None] * K + offs_k[None, :], da_tile.to(DA_ptr.dtype.element_ty), mask=mask_mk)

        acc_db += tl.dot(tl.trans(a_tile), dd_tile)

    tl.store(DB_ptr + offs_n[:, None] * K + offs_k[None, :], tl.trans(acc_db).to(DB_ptr.dtype.element_ty), mask=mask_nk)


# ============================================================================
# Fused compute_h kernels (forward: matmul_norm + compute_h, backward: 3-kernel)
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=nw)
        for bm in (32, 64, 128)
        for nw in (2, 4, 8)
    ],
    key=["M", "N"],
)
@triton.jit
def _triton_grad_h_proj_kernel(
    GRAD_Y_ptr, Y_ACT_ptr, PROJ_ptr, R_ptr, GRAD_R_EXT_ptr,
    Alpha_pre_ptr, Alpha_post_ptr, Alpha_res_ptr,
    GRAD_H_ptr, GRAD_PROJ_ptr, GRAD_R_TOTAL_ptr,
    M, N, n: tl.constexpr, eps,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Precompute grad_h, grad_proj, grad_r_total.  Grid: (cdiv(M, BLOCK_M),)."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    alpha_pre = tl.load(Alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(Alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(Alpha_res_ptr).to(tl.float32)

    # Build masks and scale
    mask_pre_n = (offs_n < n).to(tl.float32)
    mask_post_n = ((offs_n >= n) & (offs_n < 2 * n)).to(tl.float32)
    mask_res_n = (offs_n >= 2 * n).to(tl.float32)
    scale = alpha_pre * mask_pre_n + alpha_post * mask_post_n + alpha_res * mask_res_n

    # Load tiles [BLOCK_M, N_PAD] with N mask
    ptrs_mn = offs_m[:, None] * N + offs_n[None, :]
    mask_mn = mask_m[:, None] & mask_n[None, :]

    gy = tl.load(GRAD_Y_ptr + ptrs_mn, mask=mask_mn, other=0.0).to(tl.float32)
    y = tl.load(Y_ACT_ptr + ptrs_mn, mask=mask_mn, other=0.0).to(tl.float32)
    proj = tl.load(PROJ_ptr + ptrs_mn, mask=mask_mn, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)

    # Activation backward → grad_h
    gh_pre = gy * y * (1.0 - y) * mask_pre_n[None, :]
    half_y = y * 0.5
    gh_post = gy * half_y * (1.0 - half_y) * 2.0 * mask_post_n[None, :]
    gh_res = gy * mask_res_n[None, :]
    grad_h = gh_pre + gh_post + gh_res

    r_eps = r[:, None] + eps
    inv_r_eps = 1.0 / r_eps
    grad_proj = grad_h * scale[None, :] * inv_r_eps

    # grad_r_total = sum_n(grad_h * proj * scale * (-inv_r_eps^2)) + grad_r_ext
    grad_r_from_h = tl.sum(
        grad_h * proj * scale[None, :] * (-inv_r_eps * inv_r_eps), axis=1,
    )
    grad_r_ext = tl.load(GRAD_R_EXT_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    grad_r_total = grad_r_from_h + grad_r_ext

    tl.store(GRAD_H_ptr + ptrs_mn, grad_h.to(GRAD_H_ptr.dtype.element_ty), mask=mask_mn)
    tl.store(GRAD_PROJ_ptr + ptrs_mn, grad_proj.to(GRAD_PROJ_ptr.dtype.element_ty), mask=mask_mn)
    tl.store(GRAD_R_TOTAL_ptr + offs_m, grad_r_total.to(GRAD_R_TOTAL_ptr.dtype.element_ty), mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=nw)
        for bm in (32, 64, 128)
        for nw in (2, 4, 8)
    ],
    key=["M", "N"],
)
@triton.jit
def _triton_scalar_grads_kernel(
    GRAD_H_ptr, PROJ_ptr, R_ptr,
    G_ALPHA_PRE_ptr, G_ALPHA_POST_ptr, G_ALPHA_RES_ptr, G_BIAS_ptr,
    M, N, n: tl.constexpr, eps,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute scalar gradients (alpha_pre/post/res, bias).  Grid: (cdiv(M, BLOCK_M),)."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    mask_pre_n = (offs_n < n).to(tl.float32)
    mask_post_n = ((offs_n >= n) & (offs_n < 2 * n)).to(tl.float32)
    mask_res_n = (offs_n >= 2 * n).to(tl.float32)

    ptrs_mn = offs_m[:, None] * N + offs_n[None, :]
    mask_mn = mask_m[:, None] & mask_n[None, :]

    grad_h = tl.load(GRAD_H_ptr + ptrs_mn, mask=mask_mn, other=0.0).to(tl.float32)
    proj = tl.load(PROJ_ptr + ptrs_mn, mask=mask_mn, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)

    inv_r_eps = 1.0 / (r[:, None] + eps)
    ga_all = grad_h * proj * inv_r_eps

    tl.atomic_add(G_ALPHA_PRE_ptr, tl.sum(ga_all * mask_pre_n[None, :]).to(tl.float32))
    tl.atomic_add(G_ALPHA_POST_ptr, tl.sum(ga_all * mask_post_n[None, :]).to(tl.float32))
    tl.atomic_add(G_ALPHA_RES_ptr, tl.sum(ga_all * mask_res_n[None, :]).to(tl.float32))

    partial_gb = tl.sum(grad_h, axis=0)  # [N_PAD]
    tl.atomic_add(G_BIAS_ptr + offs_n, partial_gb.to(tl.float32), mask=mask_n)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=nw)
        for bm in (32, 64, 128)
        for nw in (2, 4, 8)
    ],
    key=["M", "N"],
)
@triton.jit
def _triton_compute_h_kernel(
    PROJ_ptr, SUM_SQ_ptr, Bias_ptr,
    Alpha_pre_ptr, Alpha_post_ptr, Alpha_res_ptr,
    Y_ptr, PROJ_OUT_ptr, R_ptr,
    M, N, K: tl.constexpr,
    n: tl.constexpr, eps,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute r from sum_sq, apply activations, store r and y.

    Grid: (cdiv(M, BLOCK_M),).
    Input:  proj [M, N], sum_sq [M] (= ||x||^2)
    Output: y [M, N], proj_copy [M, N], r [M] (= norm/sqrt(K), for backward)
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    alpha_pre = tl.load(Alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(Alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(Alpha_res_ptr).to(tl.float32)

    mask_pre_n = (offs_n < n).to(tl.float32)
    mask_post_n = ((offs_n >= n) & (offs_n < 2 * n)).to(tl.float32)
    mask_res_n = (offs_n >= 2 * n).to(tl.float32)
    scale = alpha_pre * mask_pre_n + alpha_post * mask_post_n + alpha_res * mask_res_n

    ptrs_mn = offs_m[:, None] * N + offs_n[None, :]
    mask_mn = mask_m[:, None] & mask_n[None, :]
    proj = tl.load(PROJ_ptr + ptrs_mn, mask=mask_mn, other=0.0).to(tl.float32)
    sum_sq = tl.load(SUM_SQ_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    # Store proj for backward
    tl.store(PROJ_OUT_ptr + ptrs_mn, proj.to(PROJ_OUT_ptr.dtype.element_ty), mask=mask_mn)

    # r = norm / sqrt(K) = sqrt(sum_sq) / sqrt(K) = sqrt(sum_sq / K)
    inv_sqrt_k = 1.0 / tl.sqrt(float(K))
    norm_val = tl.sqrt(sum_sq)
    r = norm_val * inv_sqrt_k  # this is what backward expects
    tl.store(R_ptr + offs_m, r.to(R_ptr.dtype.element_ty), mask=mask_m)

    # h = proj * scale / (r + eps) + bias
    bias_val = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    linear = proj * scale[None, :] / (r[:, None] + eps) + bias_val[None, :]

    sig = tl.sigmoid(linear)
    out = sig * mask_pre_n[None, :] + 2.0 * sig * mask_post_n[None, :] + linear * mask_res_n[None, :]

    tl.store(Y_ptr + ptrs_mn, out.to(Y_ptr.dtype.element_ty), mask=mask_mn)


def _triton_proj_rms_compute_h_fwd(x, weight, bias, alpha_pre, alpha_post, alpha_res, n, eps=1e-6):
    """Fused proj_rms + compute_h: proj_rms_fwd_kernel → compute_h_kernel."""
    M, K = x.shape
    N = weight.shape[0]
    dev = x.device

    N_PAD = max(_next_power_of_2(N), 16)
    x_c = x.contiguous()
    w_c = weight.contiguous()
    y_activated = torch.empty(M, N, dtype=x.dtype, device=dev)
    proj_out = torch.empty(M, N, dtype=x.dtype, device=dev)
    r = torch.empty(M, dtype=x.dtype, device=dev)

    split_k = 16 if K >= 16384 else 8 if K >= 8192 else 1

    if split_k == 1:
        # Kernel 1: matmul + sum_sq
        proj = torch.empty(M, N, dtype=x.dtype, device=dev)
        sum_sq = torch.empty(M, dtype=torch.float32, device=dev)
        grid1 = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _triton_matmul_norm_kernel[grid1](
            x_c, w_c, proj, sum_sq,
            M, N, K,
            x_c.stride(0), x_c.stride(1),
            w_c.stride(0), w_c.stride(1),
            N_PAD=N_PAD,
        )

        # Kernel 2: compute r = norm/sqrt(K), apply activations, store proj + r
        grid2 = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _triton_compute_h_kernel[grid2](
            proj, sum_sq, bias.contiguous(),
            alpha_pre, alpha_post, alpha_res,
            y_activated, proj_out, r,
            M, N, K, n, eps,
            N_PAD=N_PAD,
        )
    else:
        # Split-K keeps the tiny-N projection path better occupied for large
        # hidden dimensions such as n=4, C=7168.
        proj_acc = torch.empty(split_k, M, N, dtype=x.dtype, device=dev)
        sum_sq_acc = torch.empty(split_k, M, dtype=torch.float32, device=dev)
        grid1 = lambda META: (triton.cdiv(M, META["BLOCK_M"]), split_k)
        _triton_matmul_norm_split_k_kernel[grid1](
            x_c, w_c, proj_acc, sum_sq_acc,
            M, N, K,
            x_c.stride(0), x_c.stride(1),
            w_c.stride(0), w_c.stride(1),
            N_PAD=N_PAD, SPLIT_K=split_k,
        )

        grid2 = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _triton_compute_h_reduce_kernel[grid2](
            proj_acc, sum_sq_acc, bias.contiguous(),
            alpha_pre, alpha_post, alpha_res,
            y_activated, proj_out, r,
            M, N, K, n, eps,
            N_PAD=N_PAD, SPLIT_K=split_k,
        )
    return y_activated, r.unsqueeze(1), proj_out


def _triton_proj_rms_compute_h_bwd(
    grad_y, grad_r_ext, x, weight, y_activated, proj, r,
    alpha_pre, alpha_post, alpha_res, bias, n, eps=1e-6,
):
    """Backward: 3-kernel pattern matching cuTile."""
    M, K = x.shape
    N = weight.shape[0]
    N_PAD = max(_next_power_of_2(N), 16)
    dev = x.device
    dtype = x.dtype

    # 1. Precompute grad_h, grad_proj, grad_r_total
    grad_h_buf = torch.empty(M, N, dtype=torch.float32, device=dev)
    grad_proj_buf = torch.empty(M, N, dtype=torch.float32, device=dev)
    grad_r_total_buf = torch.empty(M, dtype=torch.float32, device=dev)

    r_flat = r.contiguous().view(M)
    grad_r_ext_flat = grad_r_ext.contiguous().view(M) if grad_r_ext is not None else torch.zeros(M, dtype=dtype, device=dev)

    grid1 = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    _triton_grad_h_proj_kernel[grid1](
        grad_y.contiguous(), y_activated.contiguous(), proj.contiguous(), r_flat, grad_r_ext_flat,
        alpha_pre, alpha_post, alpha_res,
        grad_h_buf, grad_proj_buf, grad_r_total_buf,
        M, N, n, eps,
        N_PAD=N_PAD,
    )

    # 2. grad_x + grad_weight (kernel reads r, computes norm = r * sqrt(K) internally)
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)

    grid2 = lambda META: (triton.cdiv(K, META["BLOCK_K"]),)
    _triton_grad_x_weight_kernel[grid2](
        x.contiguous(), weight.contiguous(), r_flat,
        grad_proj_buf, grad_r_total_buf, grad_x, grad_weight,
        M, N, K, eps,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        N_PAD=N_PAD,
    )

    # 3. Scalar gradients (alpha_pre, alpha_post, alpha_res, bias)
    g_alpha_pre = torch.zeros(1, dtype=torch.float32, device=dev)
    g_alpha_post = torch.zeros(1, dtype=torch.float32, device=dev)
    g_alpha_res = torch.zeros(1, dtype=torch.float32, device=dev)
    g_bias_acc = torch.zeros(N, dtype=torch.float32, device=dev)

    grid3 = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    _triton_scalar_grads_kernel[grid3](
        grad_h_buf, proj.contiguous(), r_flat,
        g_alpha_pre, g_alpha_post, g_alpha_res, g_bias_acc,
        M, N, n, eps,
        N_PAD=N_PAD,
    )

    return (
        grad_x,
        grad_weight,
        g_alpha_pre.to(alpha_pre.dtype),
        g_alpha_post.to(alpha_post.dtype),
        g_alpha_res.to(alpha_res.dtype),
        g_bias_acc.to(bias.dtype),
    )


# ============================================================================
# Autograd Functions + Public API
# ============================================================================


class TritonFusedSinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_logits, num_iterations, eps=1e-6):
        out, M_init = _triton_sinkhorn_fwd(input_logits, num_iterations, eps)
        ctx.save_for_backward(M_init)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (M_init,) = ctx.saved_tensors
        grad_input = _triton_sinkhorn_bwd(grad_output, M_init, ctx.num_iterations, ctx.eps)
        return grad_input, None, None


class TritonFusedHAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_pre):
        out = _triton_h_aggregate_fwd(x, h_pre)
        ctx.save_for_backward(x, h_pre)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, h_pre = ctx.saved_tensors
        return _triton_h_aggregate_bwd(grad_output, x, h_pre)


class TritonFusedHPostBDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_res, original_residual, h_post, x, bias):
        output = _triton_h_post_bda_fwd(h_res, original_residual, h_post, x, bias)
        if bias is not None:
            ctx.save_for_backward(h_res, original_residual, h_post, x, bias)
            ctx.has_bias = True
        else:
            ctx.save_for_backward(h_res, original_residual, h_post, x)
            ctx.has_bias = False
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            h_res, orig_res, h_post, x, bias = ctx.saved_tensors
        else:
            h_res, orig_res, h_post, x = ctx.saved_tensors
            bias = None
        return _triton_h_post_bda_bwd(grad_output, h_res, orig_res, h_post, x, bias)


class TritonFusedProjRmsComputeH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps=1e-6):
        y_activated, r, proj = _triton_proj_rms_compute_h_fwd(
            x, weight, bias, alpha_pre, alpha_post, alpha_res, n, eps,
        )
        ctx.save_for_backward(x, weight, y_activated, proj, r, alpha_pre, alpha_post, alpha_res, bias)
        ctx.n = n
        ctx.eps = eps
        return y_activated, r

    @staticmethod
    def backward(ctx, grad_y, grad_r):
        x, weight, y_activated, proj, r, alpha_pre, alpha_post, alpha_res, bias = ctx.saved_tensors
        g_x, g_w, g_ap, g_apo, g_ar, g_b = _triton_proj_rms_compute_h_bwd(
            grad_y, grad_r, x, weight, y_activated, proj, r,
            alpha_pre, alpha_post, alpha_res, bias, ctx.n, ctx.eps,
        )
        return g_x, g_w, g_ap, g_apo, g_ar, g_b, None, None


# --- Public API ---

def triton_fused_sinkhorn(input_logits, num_iterations, eps=1e-6):
    return TritonFusedSinkhorn.apply(input_logits, num_iterations, eps)

def triton_fused_h_aggregate(x, h_pre):
    return TritonFusedHAggregate.apply(x, h_pre)

def triton_fused_h_post_bda(h_res, original_residual, h_post, x, bias):
    return TritonFusedHPostBDA.apply(h_res, original_residual, h_post, x, bias)

def triton_fused_proj_rms(x, weight, eps=1e-6):
    """Standalone proj_rms — uses PyTorch ops (not a hot path)."""
    proj = torch.matmul(x, weight.t())
    norm = x.float().norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    r = 1.0 / (norm / math.sqrt(K) + eps)
    return proj, r.to(x.dtype)

def triton_fused_proj_rms_compute_h(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps=1e-6):
    return TritonFusedProjRmsComputeH.apply(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps)

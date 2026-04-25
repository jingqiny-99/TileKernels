import math
from collections.abc import Callable

import pytest
import torch


DTYPE = torch.bfloat16
DEVICE = 'cuda'
FWD_ATOL = 2e-2
FWD_RTOL = 2e-2
BWD_ATOL = 5e-2
BWD_RTOL = 5e-2


@pytest.fixture(autouse=True)
def _skip_without_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')


def _rand(*shape: int, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)


def _ref_sinkhorn(logits: torch.Tensor, num_iters: int, eps: float = 1e-6) -> torch.Tensor:
    row_max = logits.max(dim=-1, keepdim=True).values
    mat = torch.exp(logits - row_max)
    for _ in range(num_iters):
        mat = mat / mat.sum(dim=-1, keepdim=True).clamp(min=eps)
        mat = mat / mat.sum(dim=-2, keepdim=True).clamp(min=eps)
    return mat


def _ref_h_aggregate(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _ref_h_post_bda(
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    s, b, n, hidden = original_residual.shape
    mixed = torch.bmm(h_res.view(s * b, n, n), original_residual.float().view(s * b, n, hidden))
    mixed = mixed.view(s, b, n, hidden)
    x_bias = x.float() if bias is None else x.float() + bias.float().view(1, 1, hidden)
    return mixed + h_post.unsqueeze(-1) * x_bias.unsqueeze(2)


def _ref_proj_rms_compute_h(
    x: torch.Tensor,
    weight: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    bias: torch.Tensor,
    n: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    proj = torch.matmul(x, weight.t())
    r = x.float().norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
    scale = torch.cat(
        [
            alpha_pre.expand(n),
            alpha_post.expand(n),
            alpha_res.expand(weight.shape[0] - 2 * n),
        ],
        dim=0,
    )
    linear = proj.float() * scale.unsqueeze(0) / (r + eps) + bias.unsqueeze(0)
    return torch.cat([linear[:, :n].sigmoid(), linear[:, n : 2 * n].sigmoid() * 2, linear[:, 2 * n :]], dim=-1), r


def _load_backend(name: str) -> dict[str, Callable]:
    if name == 'tilelang':
        pytest.importorskip('tilelang')
        from tile_kernels.modeling.mhc.ops.megatron_tilelang import (
            tilelang_fused_h_aggregate,
            tilelang_fused_h_post_bda,
            tilelang_fused_sinkhorn,
        )

        return {
            'sinkhorn': tilelang_fused_sinkhorn,
            'h_aggregate': tilelang_fused_h_aggregate,
            'h_post_bda': tilelang_fused_h_post_bda,
        }
    if name == 'triton':
        pytest.importorskip('triton')
        from tile_kernels.modeling.mhc.ops.megatron_triton import (
            triton_fused_h_aggregate,
            triton_fused_h_post_bda,
            triton_fused_proj_rms_compute_h,
            triton_fused_sinkhorn,
        )

        return {
            'sinkhorn': triton_fused_sinkhorn,
            'h_aggregate': triton_fused_h_aggregate,
            'h_post_bda': triton_fused_h_post_bda,
            'proj_rms_compute_h': triton_fused_proj_rms_compute_h,
        }
    if name == 'cutile':
        from tile_kernels.modeling.mhc.ops import megatron_cutile

        if not megatron_cutile.is_cutile_available():
            pytest.skip('cuTile not available')
        return {
            'sinkhorn': megatron_cutile.fused_sinkhorn,
            'h_aggregate': megatron_cutile.fused_h_aggregate,
            'h_post_bda': megatron_cutile.fused_h_post_bda,
            'proj_rms_compute_h': megatron_cutile.fused_proj_rms_compute_h,
        }
    raise AssertionError(f'unknown backend: {name}')


@pytest.mark.parametrize('backend', ['triton', 'cutile'])
@pytest.mark.parametrize('shape,iters', [((2, 3, 4, 4), 5), ((1, 2, 4, 4), 10)])
def test_megatron_sinkhorn_fwd_bwd(backend: str, shape: tuple[int, ...], iters: int) -> None:
    fn = _load_backend(backend)['sinkhorn']
    eps = 1e-6
    data = _rand(*shape)
    grad = _rand(*shape)

    inp = data.clone().requires_grad_()
    out = fn(inp, iters, eps)
    out.backward(grad)

    ref_inp = data.clone().requires_grad_()
    ref = _ref_sinkhorn(ref_inp, iters, eps)
    ref.backward(grad)

    torch.testing.assert_close(out, ref, atol=FWD_ATOL, rtol=FWD_RTOL)
    torch.testing.assert_close(inp.grad, ref_inp.grad, atol=BWD_ATOL, rtol=BWD_RTOL)


@pytest.mark.parametrize('backend', ['tilelang', 'triton', 'cutile'])
def test_h_aggregate_matches_reference_and_preserves_mix_grad_dtype(backend: str) -> None:
    fn = _load_backend(backend)['h_aggregate']
    x_data = _rand(2, 3, 4, 128)
    h_data = _rand(2, 3, 4, dtype=torch.float32).sigmoid()
    grad = _rand(2, 3, 128)

    x = x_data.clone().requires_grad_()
    h = h_data.clone().requires_grad_()
    out = fn(x, h)
    out.backward(grad)

    x_ref = x_data.clone().requires_grad_()
    h_ref = h_data.clone().requires_grad_()
    ref = _ref_h_aggregate(x_ref, h_ref)
    ref.backward(grad)

    torch.testing.assert_close(out.float(), ref.float(), atol=FWD_ATOL, rtol=FWD_RTOL)
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), atol=BWD_ATOL, rtol=BWD_RTOL)
    torch.testing.assert_close(h.grad, h_ref.grad, atol=BWD_ATOL, rtol=BWD_RTOL)
    assert h.grad.dtype == h.dtype


@pytest.mark.parametrize('backend', ['tilelang', 'triton', 'cutile'])
@pytest.mark.parametrize('with_bias', [False, True])
def test_h_post_bda_matches_reference_and_preserves_mix_grad_dtype(backend: str, with_bias: bool) -> None:
    fn = _load_backend(backend)['h_post_bda']
    h_res_data = _rand(2, 3, 4, 4, dtype=torch.float32)
    residual_data = _rand(2, 3, 4, 128)
    h_post_data = _rand(2, 3, 4, dtype=torch.float32).sigmoid()
    x_data = _rand(2, 3, 128)
    bias_data = _rand(128) if with_bias else None
    grad = _rand(2, 3, 4, 128)

    def _run(call: Callable) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
        h_res = h_res_data.clone().requires_grad_()
        residual = residual_data.clone().requires_grad_()
        h_post = h_post_data.clone().requires_grad_()
        x = x_data.clone().requires_grad_()
        bias = bias_data.clone().requires_grad_() if bias_data is not None else None
        out = call(h_res, residual, h_post, x, bias)
        out.backward(grad)
        return out, (h_res.grad, residual.grad, h_post.grad, x.grad, bias.grad if bias is not None else None)

    out, grads = _run(fn)
    ref, ref_grads = _run(_ref_h_post_bda)

    torch.testing.assert_close(out.float(), ref.float(), atol=FWD_ATOL, rtol=FWD_RTOL)
    for actual, expected in zip(grads, ref_grads):
        if actual is None:
            assert expected is None
            continue
        torch.testing.assert_close(actual.float(), expected.float(), atol=BWD_ATOL, rtol=BWD_RTOL)
    assert grads[0].dtype == h_res_data.dtype
    assert grads[2].dtype == h_post_data.dtype


@pytest.mark.parametrize('backend', ['triton', 'cutile'])
@pytest.mark.parametrize('hidden', [128, 8192])
def test_proj_rms_compute_h_matches_reference(backend: str, hidden: int) -> None:
    fn = _load_backend(backend)['proj_rms_compute_h']
    n = 4
    out_features = n * n + 2 * n
    eps = 1e-6
    x_data = _rand(8, hidden)
    weight_data = _rand(out_features, hidden)
    alpha_pre_data = _rand(1, dtype=torch.float32)
    alpha_post_data = _rand(1, dtype=torch.float32)
    alpha_res_data = _rand(1, dtype=torch.float32)
    bias_data = _rand(out_features, dtype=torch.float32)
    grad_y = _rand(8, out_features)
    grad_r = _rand(8, 1)

    def _run(call: Callable) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
        x = x_data.clone().requires_grad_()
        weight = weight_data.clone().requires_grad_()
        alpha_pre = alpha_pre_data.clone().requires_grad_()
        alpha_post = alpha_post_data.clone().requires_grad_()
        alpha_res = alpha_res_data.clone().requires_grad_()
        bias = bias_data.clone().requires_grad_()
        y, r = call(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps)
        torch.autograd.backward([y, r], [grad_y, grad_r])
        return y, r, (x.grad, weight.grad, alpha_pre.grad, alpha_post.grad, alpha_res.grad, bias.grad)

    y, r, grads = _run(fn)
    ref_y, ref_r, ref_grads = _run(_ref_proj_rms_compute_h)

    torch.testing.assert_close(y.float(), ref_y.float(), atol=FWD_ATOL, rtol=FWD_RTOL)
    torch.testing.assert_close(r.float(), ref_r.float(), atol=FWD_ATOL, rtol=FWD_RTOL)
    for actual, expected in zip(grads, ref_grads):
        torch.testing.assert_close(actual.float(), expected.float(), atol=8e-2, rtol=2e-2)
    assert grads[2].dtype == alpha_pre_data.dtype
    assert grads[5].dtype == bias_data.dtype

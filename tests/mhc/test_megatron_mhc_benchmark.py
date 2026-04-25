from collections.abc import Callable

import pytest
import torch


DEVICE = 'cuda'
DTYPE = torch.bfloat16

_CASES = [
    (1, 1024, 4, 1280, 'tilekernels'),
    (4096, 1, 4, 7168, 'megatron'),
]


@pytest.fixture(autouse=True)
def _skip_without_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')


def _rand(*shape: int, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)


def _load_backend(name: str) -> dict[str, Callable]:
    if name == 'tilelang':
        pytest.importorskip('tilelang')
        from tile_kernels.modeling.mhc.ops.megatron_tilelang import (
            EQUIVALENCE,
            tilelang_fused_h_aggregate,
            tilelang_fused_h_post_bda,
            tilelang_fused_proj_rms_compute_h_approx,
            tilelang_fused_sinkhorn,
        )

        return {
            'equivalence': EQUIVALENCE,
            'sinkhorn': tilelang_fused_sinkhorn,
            'h_aggregate': tilelang_fused_h_aggregate,
            'h_post_bda': tilelang_fused_h_post_bda,
            'proj_rms_compute_h': tilelang_fused_proj_rms_compute_h_approx,
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
            'equivalence': {
                'sinkhorn': 'megatron-exact',
                'h_aggregate': 'megatron-exact',
                'h_post_bda': 'megatron-exact',
                'proj_rms_compute_h': 'megatron-exact',
            },
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
            'equivalence': {
                'sinkhorn': 'megatron-exact',
                'h_aggregate': 'megatron-exact',
                'h_post_bda': 'megatron-exact',
                'proj_rms_compute_h': 'megatron-exact',
            },
            'sinkhorn': megatron_cutile.fused_sinkhorn,
            'h_aggregate': megatron_cutile.fused_h_aggregate,
            'h_post_bda': megatron_cutile.fused_h_post_bda,
            'proj_rms_compute_h': megatron_cutile.fused_proj_rms_compute_h,
        }
    raise AssertionError(f'unknown backend: {name}')


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', ['tilelang', 'triton', 'cutile'])
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_sinkhorn_benchmark(
    backend: str,
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    del hidden
    backend_impl = _load_backend(backend)
    fn = backend_impl['sinkhorn']
    logits = _rand(s, b, n, n, dtype=torch.float32)
    grad = _rand(s, b, n, n, dtype=torch.float32)

    def bench_fn() -> None:
        x = logits.clone().requires_grad_()
        out = fn(x, 10, 1e-6)
        out.backward(grad)

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    benchmark_record(
        kernel='megatron_mhc_sinkhorn',
        operation=backend,
        params={'case': case_name, 's': s, 'b': b, 'n': n},
        time_us=time_us,
        extras={'equivalence': backend_impl['equivalence']['sinkhorn']},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', ['tilelang', 'triton', 'cutile'])
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_h_aggregate_benchmark(
    backend: str,
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    backend_impl = _load_backend(backend)
    fn = backend_impl['h_aggregate']
    x_data = _rand(s, b, n, hidden)
    h_data = _rand(s, b, n, dtype=torch.float32).sigmoid()
    grad = _rand(s, b, hidden)

    def bench_fn() -> None:
        x = x_data.clone().requires_grad_()
        h = h_data.clone().requires_grad_()
        out = fn(x, h)
        out.backward(grad)

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    n_tokens = s * b
    io_bytes = n_tokens * (n * hidden * 2 + n * 4 + hidden * 2)
    benchmark_record(
        kernel='megatron_mhc_h_aggregate',
        operation=backend,
        params={'case': case_name, 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        bandwidth_gbs=io_bytes / time_us / 1e3,
        extras={'equivalence': backend_impl['equivalence']['h_aggregate']},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', ['tilelang', 'triton', 'cutile'])
@pytest.mark.parametrize('with_bias', [False, True])
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_h_post_bda_benchmark(
    backend: str,
    with_bias: bool,
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    backend_impl = _load_backend(backend)
    fn = backend_impl['h_post_bda']
    h_res_data = _rand(s, b, n, n, dtype=torch.float32)
    residual_data = _rand(s, b, n, hidden)
    h_post_data = _rand(s, b, n, dtype=torch.float32).sigmoid()
    x_data = _rand(s, b, hidden)
    bias_data = _rand(hidden) if with_bias else None
    grad = _rand(s, b, n, hidden)

    def bench_fn() -> None:
        h_res = h_res_data.clone().requires_grad_()
        residual = residual_data.clone().requires_grad_()
        h_post = h_post_data.clone().requires_grad_()
        x = x_data.clone().requires_grad_()
        bias = bias_data.clone().requires_grad_() if bias_data is not None else None
        out = fn(h_res, residual, h_post, x, bias)
        out.backward(grad)

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    n_tokens = s * b
    io_bytes = n_tokens * (hidden * 2 + n * hidden * 2 * 2 + n * 4 + n * n * 4)
    if with_bias:
        io_bytes += hidden * 2
    benchmark_record(
        kernel='megatron_mhc_h_post_bda',
        operation=backend,
        params={'case': case_name, 'bias': with_bias, 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        bandwidth_gbs=io_bytes / time_us / 1e3,
        extras={'equivalence': backend_impl['equivalence']['h_post_bda']},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', ['tilelang', 'triton', 'cutile'])
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_proj_rms_compute_h_benchmark(
    backend: str,
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    backend_impl = _load_backend(backend)
    fn = backend_impl['proj_rms_compute_h']
    tokens = s * b
    k = n * hidden
    out_features = n * n + 2 * n
    x_data = _rand(tokens, k)
    weight_data = _rand(out_features, k)
    alpha_pre_data = _rand(1, dtype=torch.float32)
    alpha_post_data = _rand(1, dtype=torch.float32)
    alpha_res_data = _rand(1, dtype=torch.float32)
    bias_data = _rand(out_features, dtype=torch.float32)
    grad_y = _rand(tokens, out_features)
    grad_r = _rand(tokens, 1)

    def bench_fn() -> None:
        x = x_data.clone().requires_grad_()
        weight = weight_data.clone().requires_grad_()
        alpha_pre = alpha_pre_data.clone().requires_grad_()
        alpha_post = alpha_post_data.clone().requires_grad_()
        alpha_res = alpha_res_data.clone().requires_grad_()
        bias = bias_data.clone().requires_grad_()
        y, r = fn(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6)
        torch.autograd.backward([y, r], [grad_y, grad_r])

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    flops = 2 * tokens * out_features * k
    benchmark_record(
        kernel='megatron_mhc_proj_rms_compute_h',
        operation=backend,
        params={'case': case_name, 'tokens': tokens, 'n': n, 'hidden': hidden},
        time_us=time_us,
        extras={
            'equivalence': backend_impl['equivalence']['proj_rms_compute_h'],
            'tflops': flops / time_us / 1e6,
        },
    )

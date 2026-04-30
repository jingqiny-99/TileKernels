from collections.abc import Callable
from contextlib import contextmanager

import pytest
import torch


DEVICE = 'cuda'
DTYPE = torch.bfloat16

_CASES = [
    (4096, 1, 4, 7168, 'megatron'),
]
_KERNEL_BACKENDS = ['tilelang', 'torch_compile', 'triton', 'cutile']
_E2E_EXACT_BACKENDS = ['torch_compile', 'triton', 'cutile']
_E2E_PHASES = ['fwd', 'fwd_bwd']
_SINKHORN_REPEAT = 10
_EPS = 1e-6


@pytest.fixture(autouse=True)
def _skip_without_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')


def _rand(*shape: int, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)


def _mix_dtype(backend: str) -> torch.dtype:
    return torch.float32 if backend == 'tilelang' else DTYPE


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@contextmanager
def _nvtx_range(name: str):
    pushed = False
    try:
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_push(name)
            pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


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
    if name == 'torch_compile':
        from tile_kernels.modeling.mhc.ops.megatron_torch_compile import (
            EQUIVALENCE,
            torch_compile_fused_h_aggregate,
            torch_compile_fused_h_post_bda,
            torch_compile_fused_proj_rms_compute_h,
            torch_compile_fused_sinkhorn,
        )

        return {
            'equivalence': EQUIVALENCE,
            'sinkhorn': torch_compile_fused_sinkhorn,
            'h_aggregate': torch_compile_fused_h_aggregate,
            'h_post_bda': torch_compile_fused_h_post_bda,
            'proj_rms_compute_h': torch_compile_fused_proj_rms_compute_h,
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


def _run_exact_mhc_e2e(
    backend_impl: dict[str, Callable],
    residual: torch.Tensor,
    weight: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    bias: torch.Tensor,
    post_bias: torch.Tensor,
    n: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    s, b, _, hidden = residual.shape
    tokens = s * b
    flat_residual = residual.reshape(tokens, n * hidden)

    with _nvtx_range('mhc_e2e::proj_rms_compute_h'):
        y, _r = backend_impl['proj_rms_compute_h'](
            flat_residual,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n,
            eps,
        )
    with _nvtx_range('mhc_e2e::slice_mixes'):
        h_pre = y[:, :n].view(s, b, n)
        h_post = y[:, n : 2 * n].view(s, b, n)
        h_res_logits = y[:, 2 * n :].view(s, b, n, n)
    with _nvtx_range('mhc_e2e::sinkhorn'):
        h_res = backend_impl['sinkhorn'](h_res_logits, _SINKHORN_REPEAT, eps)
    with _nvtx_range('mhc_e2e::h_aggregate'):
        aggregated = backend_impl['h_aggregate'](residual, h_pre)
    with _nvtx_range('mhc_e2e::h_post_bda'):
        output = backend_impl['h_post_bda'](h_res, residual, h_post, aggregated, post_bias)
    return aggregated, output


def _run_tilelang_native_mhc_e2e(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    post_bias: torch.Tensor,
    n: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tile_kernels.modeling.mhc.functional import mhc_pre
    from tile_kernels.modeling.mhc.ops.post import mhc_post

    hidden = residual.shape[-1]
    with _nvtx_range('mhc_e2e::tilelang_mhc_pre'):
        layer_input, (post_mix, comb_mix) = mhc_pre(
            residual,
            fn,
            scale,
            base,
            norm_eps=eps,
            mhc_mult=n,
            post_mult_value=2.0,
            pre_eps=eps,
            sinkhorn_eps=eps,
            sinkhorn_repeat=_SINKHORN_REPEAT,
            n_splits=16,
        )
    with _nvtx_range('mhc_e2e::tilelang_mhc_post'):
        post_input = (layer_input + post_bias.view(1, 1, hidden)).contiguous()
        output = mhc_post(post_input, residual, post_mix.contiguous(), comb_mix.contiguous())
    return layer_input, output


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', _KERNEL_BACKENDS)
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
    dtype = _mix_dtype(backend)
    logits = _rand(s, b, n, n, dtype=dtype)
    grad = _rand(s, b, n, n, dtype=dtype)

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
        extras={
            'equivalence': backend_impl['equivalence']['sinkhorn'],
            'grad': 'dense',
            'dtype': str(dtype).replace('torch.', ''),
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', _KERNEL_BACKENDS)
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
    mix_dtype = _mix_dtype(backend)
    x_data = _rand(s, b, n, hidden)
    h_data = _rand(s, b, n, dtype=mix_dtype).sigmoid()
    grad = _rand(s, b, hidden)

    def bench_fn() -> None:
        x = x_data.clone().requires_grad_()
        h = h_data.clone().requires_grad_()
        out = fn(x, h)
        out.backward(grad)

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    n_tokens = s * b
    io_bytes = n_tokens * (n * hidden * 2 + n * _dtype_nbytes(mix_dtype) + hidden * 2)
    benchmark_record(
        kernel='megatron_mhc_h_aggregate',
        operation=backend,
        params={'case': case_name, 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        bandwidth_gbs=io_bytes / time_us / 1e3,
        extras={
            'equivalence': backend_impl['equivalence']['h_aggregate'],
            'grad': 'dense',
            'mix_dtype': str(mix_dtype).replace('torch.', ''),
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', _KERNEL_BACKENDS)
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_h_post_bda_benchmark(
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
    fn = backend_impl['h_post_bda']
    mix_dtype = _mix_dtype(backend)
    h_res_data = _rand(s, b, n, n, dtype=mix_dtype)
    residual_data = _rand(s, b, n, hidden)
    h_post_data = _rand(s, b, n, dtype=mix_dtype).sigmoid()
    x_data = _rand(s, b, hidden)
    bias_data = _rand(hidden)
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
    io_bytes = n_tokens * (
        hidden * 2 + n * hidden * 2 * 2 + n * _dtype_nbytes(mix_dtype) + n * n * _dtype_nbytes(mix_dtype)
    )
    io_bytes += hidden * 2
    benchmark_record(
        kernel='megatron_mhc_h_post_bda',
        operation=backend,
        params={'case': case_name, 'bias': True, 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        bandwidth_gbs=io_bytes / time_us / 1e3,
        extras={
            'equivalence': backend_impl['equivalence']['h_post_bda'],
            'grad': 'dense',
            'mix_dtype': str(mix_dtype).replace('torch.', ''),
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', _KERNEL_BACKENDS)
@pytest.mark.parametrize('with_bias', [False, True], ids=['no_bias', 'bias'])
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_h_post_bda_fwd_benchmark(
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
    mix_dtype = _mix_dtype(backend)
    h_res = _rand(s, b, n, n, dtype=mix_dtype)
    residual = _rand(s, b, n, hidden)
    h_post = _rand(s, b, n, dtype=mix_dtype).sigmoid()
    x = _rand(s, b, hidden)
    bias = _rand(hidden) if with_bias else None

    def bench_fn() -> None:
        with torch.no_grad():
            fn(h_res, residual, h_post, x, bias)

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    n_tokens = s * b
    io_bytes = n_tokens * (
        hidden * 2 + n * hidden * 2 + n * _dtype_nbytes(mix_dtype) + n * n * _dtype_nbytes(mix_dtype)
    )
    if with_bias:
        io_bytes += hidden * 2
    benchmark_record(
        kernel='megatron_mhc_h_post_bda_fwd',
        operation=backend,
        params={'case': case_name, 'bias': with_bias, 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        bandwidth_gbs=io_bytes / time_us / 1e3,
        extras={
            'equivalence': backend_impl['equivalence']['h_post_bda'],
            'mix_dtype': str(mix_dtype).replace('torch.', ''),
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_h_post_bda_megatron_unit_parity_benchmark(
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    backend_impl = _load_backend('cutile')
    fn = backend_impl['h_post_bda']

    def bench_fn() -> None:
        h_res = _rand(s, b, n, n).requires_grad_()
        residual = _rand(s, b, n, hidden).requires_grad_()
        h_post = _rand(s, b, n).requires_grad_()
        x = _rand(s, b, hidden).requires_grad_()
        out = fn(h_res, residual, h_post, x, None)
        out.sum().backward()

    bench_fn()
    time_us = benchmark_timer(bench_fn, warmup=10, rep=50)
    n_tokens = s * b
    io_bytes = n_tokens * (hidden * 2 + n * hidden * 2 * 2 + n * 2 + n * n * 2)
    benchmark_record(
        kernel='megatron_mhc_h_post_bda_unit_parity',
        operation='cutile',
        params={'case': case_name, 'bias': False, 'grad': 'sum', 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        bandwidth_gbs=io_bytes / time_us / 1e3,
        extras={
            'equivalence': 'matches Megatron-LM test_fused_mhc_kernels_bench.py',
            'mix_dtype': str(DTYPE).replace('torch.', ''),
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', _KERNEL_BACKENDS)
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
    alpha_pre_data = _rand(1)
    alpha_post_data = _rand(1)
    alpha_res_data = _rand(1)
    bias_data = _rand(out_features)
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
            'grad': 'dense',
            'param_dtype': str(DTYPE).replace('torch.', ''),
            'tflops': flops / time_us / 1e6,
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('backend', _E2E_EXACT_BACKENDS)
@pytest.mark.parametrize('phase', _E2E_PHASES)
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_mhc_e2e_exact_benchmark(
    backend: str,
    phase: str,
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    backend_impl = _load_backend(backend)
    k = n * hidden
    out_features = n * n + 2 * n
    residual_data = _rand(s, b, n, hidden)
    weight_data = _rand(out_features, k)
    alpha_pre_data = _rand(1)
    alpha_post_data = _rand(1)
    alpha_res_data = _rand(1)
    bias_data = _rand(out_features)
    post_bias_data = _rand(hidden)

    def bench_fn() -> None:
        with torch.enable_grad():
            residual = residual_data.clone().requires_grad_()
            weight = weight_data.clone().requires_grad_()
            alpha_pre = alpha_pre_data.clone().requires_grad_()
            alpha_post = alpha_post_data.clone().requires_grad_()
            alpha_res = alpha_res_data.clone().requires_grad_()
            bias = bias_data.clone().requires_grad_()
            post_bias = post_bias_data.clone().requires_grad_()
            aggregated, output = _run_exact_mhc_e2e(
                backend_impl,
                residual,
                weight,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                post_bias,
                n,
                _EPS,
            )
            if phase == 'fwd_bwd':
                with _nvtx_range('mhc_e2e::backward'):
                    (output.sum() + aggregated.sum()).backward()

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    benchmark_record(
        kernel='megatron_mhc_e2e',
        operation=backend,
        params={'case': case_name, 'phase': phase, 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        extras={
            'equivalence': 'megatron-exact-simple-pipeline',
            'fwd_grad_mode': 'enabled',
            'sinkhorn_repeat': _SINKHORN_REPEAT,
            'sublayer': 'identity',
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('phase', _E2E_PHASES)
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_mhc_e2e_tilelang_native_benchmark(
    phase: str,
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    pytest.importorskip('tilelang')
    k = n * hidden
    out_features = n * n + 2 * n
    residual_data = _rand(s, b, n, hidden)
    fn_data = _rand(out_features, k, dtype=torch.float32)
    scale_data = _rand(3, dtype=torch.float32)
    base_data = _rand(out_features, dtype=torch.float32)
    post_bias_data = _rand(hidden)

    def bench_fn() -> None:
        with torch.enable_grad():
            residual = residual_data.clone().requires_grad_()
            fn = fn_data.clone().requires_grad_()
            scale = scale_data.clone().requires_grad_()
            base = base_data.clone().requires_grad_()
            post_bias = post_bias_data.clone().requires_grad_()
            layer_input, output = _run_tilelang_native_mhc_e2e(
                residual,
                fn,
                scale,
                base,
                post_bias,
                n,
                _EPS,
            )
            if phase == 'fwd_bwd':
                with _nvtx_range('mhc_e2e::backward'):
                    (output.sum() + layer_input.sum()).backward()

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    benchmark_record(
        kernel='megatron_mhc_e2e',
        operation='tilelang',
        params={'case': case_name, 'phase': phase, 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        extras={
            'equivalence': 'tilelang-native-approx-simple-pipeline',
            'fwd_grad_mode': 'enabled',
            'mix_dtype': 'float32',
            'sinkhorn_repeat': _SINKHORN_REPEAT,
            'sublayer': 'identity',
        },
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('split_k', [4, 8, 16])
@pytest.mark.parametrize('s,b,n,hidden,case_name', _CASES)
def test_tilelang_prework_benchmark(
    split_k: int,
    s: int,
    b: int,
    n: int,
    hidden: int,
    case_name: str,
    benchmark_record,
    benchmark_timer,
) -> None:
    pytest.importorskip('tilelang')
    from tile_kernels.modeling.mhc.ops.pre_big_fuse import mhc_pre_big_fuse

    tokens = s * b
    out_features = n * n + 2 * n
    k = n * hidden
    residual_data = _rand(s, b, n, hidden)
    fn_data = _rand(out_features, k, dtype=torch.float32)
    scale_data = _rand(3, dtype=torch.float32)
    base_data = _rand(out_features, dtype=torch.float32)

    def bench_fn() -> None:
        post_mix, comb_mix, layer_input = mhc_pre_big_fuse(
            residual_data,
            fn_data,
            scale_data,
            base_data,
            rms_eps=1e-6,
            mhc_pre_eps=1e-6,
            mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=10,
            n_splits=split_k,
        )
        assert post_mix.shape == (s, b, n, 1)
        assert comb_mix.shape == (s, b, n, n)
        assert layer_input.shape == (s, b, hidden)

    bench_fn()
    time_us = benchmark_timer(bench_fn)
    flops = 2 * tokens * out_features * k
    benchmark_record(
        kernel='megatron_mhc_prework_fwd',
        operation='tilelang',
        params={'case': case_name, 'tokens': tokens, 'n': n, 'hidden': hidden, 'split_k': split_k},
        time_us=time_us,
        extras={
            'equivalence': 'approximate-fused-forward',
            'includes': 'proj+mapping+sinkhorn+h_aggregate',
            'split_k': split_k,
            'tflops': flops / time_us / 1e6,
        },
    )

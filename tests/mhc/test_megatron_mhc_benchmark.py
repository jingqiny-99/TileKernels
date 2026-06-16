from collections.abc import Callable
from contextlib import contextmanager
import os

import pytest
import torch


DEVICE = 'cuda'
DTYPE = torch.bfloat16

def _int_list_from_env(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    values = [int(part.strip()) for part in value.split(',') if part.strip()]
    if not values:
        raise ValueError(f'{name} must contain at least one integer')
    if any(item <= 0 for item in values):
        raise ValueError(f'{name} must contain positive integers')
    return values


_SEQLENS = _int_list_from_env('MHC_BENCH_SEQLENS', [1024, 2048, 4096, 8192])
_BATCH_SIZES = _int_list_from_env('MHC_BENCH_BATCH_SIZES', [1, 4])
_HIDDENS = _int_list_from_env('MHC_BENCH_HIDDENS', [4096, 7168])
_CASES = [
    (s, b, 4, hidden, f's{s}_b{b}_h{hidden}')
    for hidden in _HIDDENS
    for b in _BATCH_SIZES
    for s in _SEQLENS
]
_ALL_KERNEL_BACKENDS = ['tilelang', 'megatron_lm', 'mhc_bench_triton']
_ALL_E2E_EXACT_BACKENDS = ['megatron_lm', 'mhc_bench_triton']
_SOURCE_ALIASES = {
    'all': _ALL_KERNEL_BACKENDS,
    'tilelang': ['tilelang'],
    'tl': ['tilelang'],
    'megatron': ['megatron_lm'],
    'megatron_lm': ['megatron_lm'],
    'megatron-lm': ['megatron_lm'],
    'mhc_bench': ['mhc_bench_triton'],
    'mhc-bench': ['mhc_bench_triton'],
    'mhc_bench_triton': ['mhc_bench_triton'],
    'triton': ['mhc_bench_triton'],
}
_ALL_SCOPES = {
    'sinkhorn',
    'h_aggregate',
    'h_post_bda',
    'h_post_bda_fwd',
    'proj_rms_compute_h',
    'mhc_e2e',
    'tilelang_e2e',
    'tilelang_prework',
    'megatron_unit_parity',
}
_SCOPE_ALIASES = {
    'all': sorted(_ALL_SCOPES),
    'kernels': [
        'sinkhorn',
        'h_aggregate',
        'h_post_bda',
        'h_post_bda_fwd',
        'proj_rms_compute_h',
        'megatron_unit_parity',
    ],
    'kernel': [
        'sinkhorn',
        'h_aggregate',
        'h_post_bda',
        'h_post_bda_fwd',
        'proj_rms_compute_h',
        'megatron_unit_parity',
    ],
    'e2e': ['mhc_e2e', 'tilelang_e2e'],
    'prework': ['tilelang_prework'],
    **{scope: [scope] for scope in _ALL_SCOPES},
}
_KERNEL_BACKENDS: list[str]
_E2E_EXACT_BACKENDS: list[str]
_E2E_PHASES = ['fwd', 'fwd_bwd']
_SINKHORN_REPEAT = 10
_EPS = 1e-6


def _name_list_from_env(name: str, default: str, aliases: dict[str, list[str]]) -> list[str]:
    value = os.environ.get(name, default)
    selected: list[str] = []
    for raw_part in value.split(','):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part not in aliases:
            valid = ', '.join(sorted(aliases))
            raise ValueError(f'{name} contains unsupported value {part!r}; expected one of: {valid}')
        for item in aliases[part]:
            if item not in selected:
                selected.append(item)
    if not selected:
        raise ValueError(f'{name} must select at least one value')
    return selected


_SELECTED_SOURCES = set(_name_list_from_env('MHC_BENCH_SOURCES', 'all', _SOURCE_ALIASES))
_SELECTED_SCOPES = set(_name_list_from_env('MHC_BENCH_SCOPES', 'all', _SCOPE_ALIASES))
_KERNEL_BACKENDS = [backend for backend in _ALL_KERNEL_BACKENDS if backend in _SELECTED_SOURCES]
_E2E_EXACT_BACKENDS = [backend for backend in _ALL_E2E_EXACT_BACKENDS if backend in _SELECTED_SOURCES]


def _scope_enabled(*scopes: str) -> bool:
    return bool(_SELECTED_SCOPES.intersection(scopes))


def _source_enabled(*sources: str) -> bool:
    return bool(_SELECTED_SOURCES.intersection(sources))


def _skip_unless_scope(*scopes: str) -> None:
    if not _scope_enabled(*scopes):
        pytest.skip(f'MHC benchmark scope disabled: {",".join(scopes)}')


def _skip_unless_source(*sources: str) -> None:
    if not _source_enabled(*sources):
        pytest.skip(f'MHC benchmark source disabled: {",".join(sources)}')


@pytest.fixture(autouse=True)
def _skip_without_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')


def _rand(*shape: int, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)


def _mix_dtype(backend: str) -> torch.dtype:
    return torch.float32 if backend == 'tilelang' else DTYPE


def _is_mhc_bench_triton(backend: str) -> bool:
    return backend == 'mhc_bench_triton'


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _split_proj_rms_compute_h_result(
    result: tuple[torch.Tensor, ...],
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(result) == 4:
        h_pre, h_post, h_res, r = result
        return h_pre, h_post, h_res, r
    if len(result) == 2:
        y, r = result
        return y[:, :n], y[:, n : 2 * n], y[:, 2 * n :], r
    raise AssertionError(f'proj_rms_compute_h returned {len(result)} tensors; expected 2 or 4')


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
            'backend_detail': 'TileKernels TileLang kernels',
            'sinkhorn': tilelang_fused_sinkhorn,
            'h_aggregate': tilelang_fused_h_aggregate,
            'h_post_bda': tilelang_fused_h_post_bda,
            'proj_rms_compute_h': tilelang_fused_proj_rms_compute_h_approx,
        }
    if name == 'megatron_lm':
        from tile_kernels.modeling.mhc.ops import megatron_lm_fused

        try:
            megatron_lm_fused.require_available()
        except RuntimeError as exc:
            pytest.skip(str(exc))
        return {
            'equivalence': megatron_lm_fused.EQUIVALENCE,
            'backend_detail': megatron_lm_fused.backend_selection(),
            'sinkhorn': megatron_lm_fused.fused_sinkhorn,
            'h_aggregate': megatron_lm_fused.fused_h_aggregate,
            'h_post_bda': megatron_lm_fused.fused_h_post_bda,
            'proj_rms_compute_h': megatron_lm_fused.fused_proj_rms_compute_h,
        }
    if name == 'mhc_bench_triton':
        pytest.importorskip('triton')
        from tile_kernels.modeling.mhc.ops.mhc_bench_triton import (
            EQUIVALENCE,
            mhc_bench_triton_fused_proj_rms_compute_h,
            mhc_bench_triton_fused_sinkhorn,
            mhc_bench_triton_native_h_aggregate,
            mhc_bench_triton_native_h_post_bda,
        )

        return {
            'equivalence': EQUIVALENCE,
            'backend_detail': 'mhc_bench/triton_kernels native wrappers',
            'sinkhorn': mhc_bench_triton_fused_sinkhorn,
            'h_aggregate': mhc_bench_triton_native_h_aggregate,
            'h_post_bda': mhc_bench_triton_native_h_post_bda,
            'proj_rms_compute_h': mhc_bench_triton_fused_proj_rms_compute_h,
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
        proj_result = backend_impl['proj_rms_compute_h'](
            flat_residual,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n,
            eps,
        )
    with _nvtx_range('mhc_e2e::split_mixes'):
        h_pre_flat, h_post_flat, h_res_flat, _r = _split_proj_rms_compute_h_result(proj_result, n)
        h_pre = h_pre_flat.view(s, b, n)
        h_post = h_post_flat.view(s, b, n)
        h_res_logits = h_res_flat.view(s, b, n, n)
    with _nvtx_range('mhc_e2e::sinkhorn'):
        h_res = backend_impl['sinkhorn'](h_res_logits, _SINKHORN_REPEAT, eps)
    with _nvtx_range('mhc_e2e::h_aggregate'):
        aggregated = backend_impl['h_aggregate'](residual, h_pre)
    with _nvtx_range('mhc_e2e::h_post_bda'):
        output = backend_impl['h_post_bda'](h_res, residual, h_post, aggregated, post_bias)
    return aggregated, output


def _run_mhc_bench_triton_native_mhc_e2e(
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
    s, b, hidden, _ = residual.shape
    tokens = s * b
    flat_residual = residual.reshape(tokens, hidden * n)

    with _nvtx_range('mhc_e2e::proj_rms_compute_h'):
        proj_result = backend_impl['proj_rms_compute_h'](
            flat_residual,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n,
            eps,
        )
    with _nvtx_range('mhc_e2e::split_mixes'):
        h_pre_flat, h_post_flat, h_res_flat, _r = _split_proj_rms_compute_h_result(proj_result, n)
        h_pre = h_pre_flat.view(s, b, n)
        h_post = h_post_flat.view(s, b, n)
        h_res_logits = h_res_flat.view(s, b, n, n)
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
    _skip_unless_scope('sinkhorn')
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
            'backend_detail': backend_impl['backend_detail'],
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
    _skip_unless_scope('h_aggregate')
    backend_impl = _load_backend(backend)
    fn = backend_impl['h_aggregate']
    mix_dtype = _mix_dtype(backend)
    if _is_mhc_bench_triton(backend):
        x_data = _rand(s, b, hidden, n)
    else:
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
            'backend_detail': backend_impl['backend_detail'],
            'grad': 'dense',
            'layout': 's,b,C,n' if _is_mhc_bench_triton(backend) else 's,b,n,C',
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
    _skip_unless_scope('h_post_bda')
    backend_impl = _load_backend(backend)
    fn = backend_impl['h_post_bda']
    mix_dtype = _mix_dtype(backend)
    h_res_data = _rand(s, b, n, n, dtype=mix_dtype)
    if _is_mhc_bench_triton(backend):
        residual_data = _rand(s, b, hidden, n)
        grad = _rand(s, b, hidden, n)
    else:
        residual_data = _rand(s, b, n, hidden)
        grad = _rand(s, b, n, hidden)
    h_post_data = _rand(s, b, n, dtype=mix_dtype).sigmoid()
    x_data = _rand(s, b, hidden)
    bias_data = _rand(hidden)

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
            'backend_detail': backend_impl['backend_detail'],
            'grad': 'dense',
            'layout': 's,b,C,n' if _is_mhc_bench_triton(backend) else 's,b,n,C',
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
    _skip_unless_scope('h_post_bda_fwd')
    backend_impl = _load_backend(backend)
    fn = backend_impl['h_post_bda']
    mix_dtype = _mix_dtype(backend)
    h_res = _rand(s, b, n, n, dtype=mix_dtype)
    residual = _rand(s, b, hidden, n) if _is_mhc_bench_triton(backend) else _rand(s, b, n, hidden)
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
            'backend_detail': backend_impl['backend_detail'],
            'layout': 's,b,C,n' if _is_mhc_bench_triton(backend) else 's,b,n,C',
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
    _skip_unless_source('megatron_lm')
    _skip_unless_scope('megatron_unit_parity')
    backend_impl = _load_backend('megatron_lm')
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
        operation='megatron_lm',
        params={'case': case_name, 'bias': False, 'grad': 'sum', 's': s, 'b': b, 'n': n, 'hidden': hidden},
        time_us=time_us,
        bandwidth_gbs=io_bytes / time_us / 1e3,
        extras={
            'equivalence': 'matches Megatron-LM test_fused_mhc_kernels_bench.py',
            'backend_detail': backend_impl['backend_detail'],
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
    _skip_unless_scope('proj_rms_compute_h')
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
        result = fn(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6)
        if len(result) == 4:
            h_pre, h_post, h_res, r = result
            torch.autograd.backward(
                [h_pre, h_post, h_res, r],
                [grad_y[:, :n], grad_y[:, n : 2 * n], grad_y[:, 2 * n :], grad_r],
            )
        else:
            y, r = result
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
            'backend_detail': backend_impl['backend_detail'],
            'grad': 'dense',
            'includes': 'projection+scale' if _is_mhc_bench_triton(backend) else 'proj_rms_compute_h',
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
    _skip_unless_scope('mhc_e2e')
    backend_impl = _load_backend(backend)
    k = n * hidden
    out_features = n * n + 2 * n
    if _is_mhc_bench_triton(backend):
        residual_data = _rand(s, b, hidden, n)
    else:
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
            if _is_mhc_bench_triton(backend):
                aggregated, output = _run_mhc_bench_triton_native_mhc_e2e(
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
            else:
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
            'backend_detail': backend_impl['backend_detail'],
            'fwd_grad_mode': 'enabled',
            'layout': 's,b,C,n' if _is_mhc_bench_triton(backend) else 's,b,n,C',
            'pre_path': 'projection+scale' if _is_mhc_bench_triton(backend) else 'proj_rms_compute_h',
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
    _skip_unless_source('tilelang')
    _skip_unless_scope('tilelang_e2e')
    pytest.importorskip('tilelang')
    k = n * hidden
    out_features = n * n + 2 * n
    residual_data = _rand(s, b, n, hidden)
    fn_data = _rand(out_features, k, dtype=torch.float32)
    scale_data = _rand(3, dtype=torch.float32)
    base_data = _rand(out_features, dtype=torch.float32)
    post_bias_data = _rand(hidden)

    def bench_fn() -> None:
        if phase == 'fwd':
            with torch.no_grad():
                residual = residual_data.clone()
                fn = fn_data.clone()
                scale = scale_data.clone()
                base = base_data.clone()
                post_bias = post_bias_data.clone()
                _run_tilelang_native_mhc_e2e(
                    residual,
                    fn,
                    scale,
                    base,
                    post_bias,
                    n,
                    _EPS,
                )
            return

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
            'fwd_grad_mode': 'disabled' if phase == 'fwd' else 'enabled',
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
    _skip_unless_source('tilelang')
    _skip_unless_scope('tilelang_prework')
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

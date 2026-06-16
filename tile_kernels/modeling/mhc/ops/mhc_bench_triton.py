"""Adapters for the Triton mHC kernels from the sibling ``mhc_bench`` repo.

The benchmark-facing helpers keep the mhc_bench native layout so Nsight
timelines measure kernel work instead of layout conversion. The compat helpers
present the existing Megatron-compatible TileKernels signatures for correctness
tests.
"""

from pathlib import Path
import os
import sys
from types import ModuleType

import torch


EQUIVALENCE = {
    'sinkhorn': 'mhc_bench-triton-logspace',
    'h_aggregate': 'megatron-exact; native layout is (s,b,C,n)',
    'h_post_bda': 'megatron-exact; native layout is (s,b,C,n)',
    'proj_rms_compute_h': 'projection+scale composition from mhc_bench Triton kernels',
}

_MHC_BENCH_OPS: ModuleType | None = None


def _mhc_bench_root() -> Path:
    configured = os.environ.get('MHC_BENCH_PATH')
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[5] / 'mhc_bench'


def _load_mhc_bench_ops() -> ModuleType:
    global _MHC_BENCH_OPS
    if _MHC_BENCH_OPS is not None:
        return _MHC_BENCH_OPS

    root = _mhc_bench_root()
    if not (root / 'triton_kernels' / 'mhc_ops.py').exists():
        raise RuntimeError(
            f'mhc_bench Triton kernels were not found at {root}. '
            'Set MHC_BENCH_PATH to the mhc_bench checkout.'
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from triton_kernels import mhc_ops

    _MHC_BENCH_OPS = mhc_ops
    return mhc_ops


def mhc_bench_triton_fused_sinkhorn(
    input_logits: torch.Tensor,
    num_iterations: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    del eps
    ops = _load_mhc_bench_ops()
    n = input_logits.shape[-1]
    return ops.mhc_fused_sinkhorn(input_logits, n=n, recompute_hist=True, iters=num_iterations)


def mhc_bench_triton_fused_proj_rms_compute_h(
    x: torch.Tensor,
    weight: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    del eps
    ops = _load_mhc_bench_ops()
    alpha = torch.cat(
        [
            alpha_pre.reshape(1),
            alpha_post.reshape(1),
            alpha_res.reshape(1),
        ],
        dim=0,
    )
    proj, ms = ops.mhc_fused_projection(x, weight)
    h_pre, h_post, h_res = ops.mhc_fused_scale(proj, alpha, bias.view(1, -1), ms, n)
    y = torch.cat([h_pre, h_post, h_res], dim=-1)
    r = torch.sqrt(ms).view(-1, 1).to(dtype=x.dtype)
    return y, r


def mhc_bench_triton_native_h_aggregate(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """Run mhc_bench aggregate on native ``(s, b, C, n)`` inputs."""
    ops = _load_mhc_bench_ops()
    return ops.mhc_fused_aggregate(x, h_pre, x.shape[-1])


def mhc_bench_triton_native_h_post_bda(
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run mhc_bench expand-combine on native ``(s, b, C, n)`` inputs."""
    ops = _load_mhc_bench_ops()
    return ops.mhc_fused_expand_combine(x, bias, h_post, original_residual, h_res, h_res.shape[-1], True)


def mhc_bench_triton_fused_h_aggregate(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """Megatron-compatible aggregate for correctness tests.

    Input ``x`` is ``(s, b, n, C)`` and is converted to the mhc_bench native
    ``(s, b, C, n)`` layout outside benchmark timing.
    """
    return mhc_bench_triton_native_h_aggregate(x.transpose(-1, -2).contiguous(), h_pre)


def mhc_bench_triton_fused_h_post_bda(
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Megatron-compatible post op for correctness tests."""
    output = mhc_bench_triton_native_h_post_bda(
        h_res,
        original_residual.transpose(-1, -2).contiguous(),
        h_post,
        x,
        bias,
    )
    return output.transpose(-1, -2).contiguous()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adapter for the mHC kernels shipped by Megatron-LM.

The benchmark should measure Megatron-LM's public fused mHC dispatch path, not a
TileKernels-local approximation.  This loader finds a neighboring Megatron-LM
checkout, imports ``megatron.core.fusions.fused_mhc_kernels``, and delegates to
its public functions.  Megatron-LM owns the per-op priority:

  - sinkhorn: Triton -> cuTile -> native
  - h_aggregate: fwd Triton -> cuTile -> native, bwd cuTile -> native
  - h_post_bda: Triton -> cuTile -> native
  - proj_rms_compute_h: cuTile -> native
"""

from __future__ import annotations

from pathlib import Path
import importlib
import os
import sys
from types import ModuleType

import torch


EQUIVALENCE = {
    'sinkhorn': 'megatron-lm-public-fused-dispatch',
    'h_aggregate': 'megatron-lm-public-fused-dispatch',
    'h_post_bda': 'megatron-lm-public-fused-dispatch',
    'proj_rms_compute_h': 'megatron-lm-public-fused-dispatch',
}

_MEGATRON_MHC_OPS: ModuleType | None = None
_MEGATRON_LM_ROOT: Path | None = None


def _is_megatron_lm_root(path: Path) -> bool:
    return (path / 'megatron' / 'core' / 'fusions' / 'fused_mhc_kernels.py').exists()


def _megatron_lm_root() -> Path | None:
    configured = os.environ.get('MEGATRON_LM_PATH')
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _is_megatron_lm_root(root):
            raise RuntimeError(
                f'MEGATRON_LM_PATH={root} does not contain '
                'megatron/core/fusions/fused_mhc_kernels.py.'
            )
        return root

    this_file = Path(__file__).resolve()
    candidates = [
        this_file.parents[5] / 'Megatron-LM',
        this_file.parents[5] / 'megatron-lm',
        Path.cwd().parent / 'Megatron-LM',
        Path.cwd().parent / 'megatron-lm',
        Path.cwd() / 'Megatron-LM',
        Path.cwd() / 'megatron-lm',
    ]
    for root in candidates:
        if _is_megatron_lm_root(root):
            return root.resolve()
    return None


def _load_megatron_mhc_ops() -> ModuleType:
    global _MEGATRON_MHC_OPS, _MEGATRON_LM_ROOT
    if _MEGATRON_MHC_OPS is not None:
        return _MEGATRON_MHC_OPS

    root = _megatron_lm_root()
    if root is None:
        raise RuntimeError(
            'Megatron-LM fused mHC kernels were not found. Set MEGATRON_LM_PATH '
            'to the Megatron-LM checkout.'
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    mhc_ops = importlib.import_module('megatron.core.fusions.fused_mhc_kernels')
    module_file = Path(getattr(mhc_ops, '__file__', '')).resolve()
    if root not in module_file.parents:
        raise RuntimeError(
            'Imported megatron.core.fusions.fused_mhc_kernels from '
            f'{module_file}, expected it under {root}. Start a fresh process '
            'or set MEGATRON_LM_PATH before importing Megatron.'
        )
    _MEGATRON_MHC_OPS = mhc_ops
    _MEGATRON_LM_ROOT = root
    return _MEGATRON_MHC_OPS


def require_available() -> None:
    _load_megatron_mhc_ops()


def backend_selection() -> str:
    ops = _load_megatron_mhc_ops()
    root = str(_MEGATRON_LM_ROOT) if _MEGATRON_LM_ROOT is not None else 'unknown'
    selection_fn = getattr(ops, '_mhc_backend_selection', None)
    if selection_fn is None:
        return f'Megatron-LM public fused mHC kernels from {root}'
    return f'Megatron-LM public fused mHC kernels from {root}; {selection_fn()}'


def is_triton_available() -> bool:
    ops = _load_megatron_mhc_ops()
    return bool(ops.is_triton_available())


def is_cutile_available() -> bool:
    ops = _load_megatron_mhc_ops()
    return bool(ops.is_cutile_available())


def fused_sinkhorn(input_logits: torch.Tensor, num_iterations: int, eps: float = 1e-6) -> torch.Tensor:
    ops = _load_megatron_mhc_ops()
    return ops.fused_sinkhorn(input_logits, num_iterations, eps)


def fused_h_aggregate(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    ops = _load_megatron_mhc_ops()
    return ops.fused_h_aggregate(x, h_pre)


def fused_h_post_bda(
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    ops = _load_megatron_mhc_ops()
    return ops.fused_h_post_bda(h_res, original_residual, h_post, x, bias)


def fused_proj_rms_compute_h(
    x: torch.Tensor,
    weight: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ops = _load_megatron_mhc_ops()
    return ops.fused_proj_rms_compute_h(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps)

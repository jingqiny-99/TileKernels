"""TileLang adapters with Megatron-compatible MHC call signatures.

Only exact or near-exact semantic matches should be used for direct backend
comparisons.  Projection/compute_h is exposed as an approximate adapter so the
pipeline benchmark can report it without implying strict numerical equivalence.
"""

import torch

from .norm_fn import mhc_pre_norm_fn
from .post import mhc_post
from .pre_apply_mix import mhc_pre_apply_mix
from .sinkhorn import sinkhorn_normalize


EQUIVALENCE = {
    'sinkhorn': 'near-equivalent',
    'h_aggregate': 'exact',
    'h_post_bda': 'exact when bias is folded into x and h_res is transposed for mhc_post',
    'proj_rms_compute_h': 'approximate',
}


def tilelang_fused_sinkhorn(input_logits: torch.Tensor, num_iterations: int, eps: float = 1e-6) -> torch.Tensor:
    return sinkhorn_normalize(input_logits, repeat=num_iterations, eps=eps)


def tilelang_fused_h_aggregate(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    return mhc_pre_apply_mix(x, h_pre.unsqueeze(-1))


def tilelang_fused_h_post_bda(
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if bias is not None:
        x = x + bias.view(*((1,) * (x.ndim - 1)), bias.shape[-1])
    return mhc_post(x, original_residual, h_post.unsqueeze(-1), h_res.transpose(-1, -2).contiguous())


def tilelang_fused_proj_rms_compute_h_approx(
    x: torch.Tensor,
    weight: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate Megatron proj_rms_compute_h through TileLang pre_norm_fn.

    TileLang's pre_norm_fn uses rsqrt(mean(x^2) + eps), while Megatron's fused
    path uses 1 / (sqrt(mean(x^2)) + eps).  Keep this out of exact correctness
    comparisons.
    """
    mixes = mhc_pre_norm_fn(x.unsqueeze(-2), weight.float(), None, eps, fuse_grad_acc=False)
    scale = torch.cat(
        [
            alpha_pre.expand(n),
            alpha_post.expand(n),
            alpha_res.expand(weight.shape[0] - 2 * n),
        ],
        dim=0,
    )
    linear = mixes * scale.unsqueeze(0) + bias.unsqueeze(0)
    pre = linear[:, :n].sigmoid()
    post = linear[:, n : 2 * n].sigmoid() * 2
    out = torch.cat([pre, post, linear[:, 2 * n :]], dim=-1)
    r = x.float().norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return out.to(x.dtype), r.to(x.dtype)

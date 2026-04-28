"""TileLang adapters with Megatron-compatible MHC call signatures.

Only exact or near-exact semantic matches should be used for direct backend
comparisons.  Projection/compute_h is exposed as an approximate adapter so the
pipeline benchmark can report it without implying strict numerical equivalence.
"""

import torch

from tile_kernels.mhc.norm_fn_kernel import _mhc_pre_norm_fn_bwd_mul, _mhc_pre_norm_fn_fwd_mul, round_to_tf32

from .post import mhc_post
from .pre_apply_mix import mhc_pre_apply_mix
from .sinkhorn import sinkhorn_normalize


EQUIVALENCE = {
    'sinkhorn': 'near-equivalent',
    'h_aggregate': 'exact',
    'h_post_bda': 'exact when bias is folded into x and h_res is transposed for mhc_post',
    'proj_rms_compute_h': 'near-exact formula, TileLang TF32 projection, no split-K',
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


class TileLangFusedProjRmsComputeHBetterApprox(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        bias: torch.Tensor,
        n: int,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Megatron-formula adapter backed by TileLang raw projection kernels.

        This keeps Megatron's ``proj * scale / (r + eps) + bias`` semantics, but
        still uses the existing TileLang GEMM path, which has no split-K support.
        """
        assert x.ndim == 2
        assert weight.ndim == 2
        assert x.dtype == torch.bfloat16

        m, k = x.shape
        out_features = weight.shape[0]
        assert weight.shape[1] == k
        assert out_features == n * n + 2 * n == 24

        x_c = x.contiguous()
        fn = round_to_tf32(weight.float().contiguous())

        proj_buf = torch.empty(m, 1, out_features, dtype=torch.float32, device=x.device)
        sqrsum_buf = torch.empty(m, 1, dtype=torch.float32, device=x.device)
        _mhc_pre_norm_fn_fwd_mul(out_features, 1, k)(
            x_c,
            fn,
            proj_buf,
            sqrsum_buf,
        )

        proj = proj_buf.view(m, out_features)
        r = torch.sqrt(sqrsum_buf.view(m, 1) / k)
        scale = _make_mhc_scale(alpha_pre, alpha_post, alpha_res, out_features, n)
        linear = proj * scale.unsqueeze(0) / (r + eps) + bias.float().unsqueeze(0)
        y = _apply_mhc_activations(linear, n)

        ctx.save_for_backward(x_c, fn, proj, r, y.to(x.dtype), alpha_pre, alpha_post, alpha_res)
        ctx.n = n
        ctx.eps = eps
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype

        return y.to(x.dtype), r.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_r_ext: torch.Tensor | None):
        x, fn, proj, r, y, alpha_pre, alpha_post, alpha_res = ctx.saved_tensors
        n = ctx.n
        eps = ctx.eps
        m, k = x.shape
        out_features = proj.shape[1]

        grad_h = _compute_mhc_activation_grad(grad_y.float(), y.float(), n)
        scale = _make_mhc_scale(alpha_pre, alpha_post, alpha_res, out_features, n)
        inv_r_eps = 1.0 / (r + eps)
        grad_proj = grad_h * scale.unsqueeze(0) * inv_r_eps

        grad_r_from_h = torch.sum(
            grad_h * proj * scale.unsqueeze(0) * (-(inv_r_eps * inv_r_eps)),
            dim=1,
            keepdim=True,
        )
        if grad_r_ext is None:
            grad_r_total = grad_r_from_h
        else:
            grad_r_total = grad_r_from_h + grad_r_ext.float()
        sqrsum_grad = grad_r_total / (2.0 * k * r)

        grad_x = torch.zeros_like(x)
        grad_weight = torch.empty_like(fn)
        _mhc_pre_norm_fn_bwd_mul(out_features, 1, k)(
            grad_proj.view(m, 1, out_features),
            sqrsum_grad.view(m, 1),
            x,
            fn,
            grad_x,
            grad_weight,
        )

        grad_alpha_all = grad_h * proj * inv_r_eps
        grad_alpha_pre = grad_alpha_all[:, :n].sum().reshape_as(alpha_pre).to(alpha_pre.dtype)
        grad_alpha_post = grad_alpha_all[:, n : 2 * n].sum().reshape_as(alpha_post).to(alpha_post.dtype)
        grad_alpha_res = grad_alpha_all[:, 2 * n :].sum().reshape_as(alpha_res).to(alpha_res.dtype)
        grad_bias = grad_h.sum(dim=0).to(ctx.bias_dtype)

        return (
            grad_x,
            grad_weight.to(ctx.weight_dtype),
            grad_alpha_pre,
            grad_alpha_post,
            grad_alpha_res,
            grad_bias,
            None,
            None,
        )


def _make_mhc_scale(
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    out_features: int,
    n: int,
) -> torch.Tensor:
    scale = torch.empty(out_features, dtype=torch.float32, device=alpha_pre.device)
    scale[:n] = alpha_pre.float().reshape(1).expand(n)
    scale[n : 2 * n] = alpha_post.float().reshape(1).expand(n)
    scale[2 * n :] = alpha_res.float().reshape(1).expand(out_features - 2 * n)
    return scale


def _apply_mhc_activations(linear: torch.Tensor, n: int) -> torch.Tensor:
    return torch.cat(
        [
            linear[:, :n].sigmoid(),
            linear[:, n : 2 * n].sigmoid() * 2,
            linear[:, 2 * n :],
        ],
        dim=-1,
    )


def _compute_mhc_activation_grad(grad_y: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    grad_h = torch.empty_like(grad_y, dtype=torch.float32)
    grad_h[:, :n] = grad_y[:, :n] * y[:, :n] * (1 - y[:, :n])
    y_post_half = y[:, n : 2 * n] * 0.5
    grad_h[:, n : 2 * n] = grad_y[:, n : 2 * n] * y_post_half * (1 - y_post_half) * 2
    grad_h[:, 2 * n :] = grad_y[:, 2 * n :]
    return grad_h


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
    return TileLangFusedProjRmsComputeHBetterApprox.apply(
        x,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        eps,
    )

"""torch.compile adapters with Megatron-compatible MHC call signatures."""

from typing import Optional, Tuple

import torch
from torch import Tensor


EQUIVALENCE = {
    'sinkhorn': 'megatron-native-torch.compile',
    'h_aggregate': 'megatron-native-torch.compile',
    'h_post_bda': 'megatron-native-torch.compile',
    'proj_rms_compute_h': 'megatron-formula-torch.compile',
}


@torch.compile
def _sinkhorn_iterations(input_logits: Tensor, num_iterations: int, eps: float) -> Tensor:
    row_max = input_logits.max(dim=-1, keepdim=True).values
    matrix = torch.exp(input_logits - row_max)
    for _ in range(num_iterations):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp(min=eps)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp(min=eps)
    return matrix


class TorchCompileSinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
        output = _sinkhorn_iterations(input_logits, num_iterations, eps)
        ctx.save_for_backward(input_logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        (input_logits,) = ctx.saved_tensors
        with torch.enable_grad():
            logits = input_logits.detach().requires_grad_(True)
            output = _sinkhorn_iterations(logits, ctx.num_iterations, ctx.eps)
            output.backward(grad_output)
        return logits.grad, None, None


def torch_compile_fused_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    return TorchCompileSinkhorn.apply(input_logits, num_iterations, eps)


@torch.compile
def torch_compile_fused_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


@torch.compile
def torch_compile_fused_h_post_bda(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    s, b, n, hidden = original_residual.shape
    h_res_batched = h_res.view(s * b, n, n)
    residual_batched = original_residual.view(s * b, n, hidden)
    mixed = torch.bmm(h_res_batched, residual_batched).view(s, b, n, hidden)
    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)
    if bias is not None:
        bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, hidden)
        return x_expanded + bias_expanded + mixed
    return x_expanded + mixed


@torch.compile
def torch_compile_fused_proj_rms_compute_h(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    proj = torch.matmul(x, weight.t())
    x_float = x.float()
    r = torch.sqrt(torch.sum(x_float * x_float, dim=-1, keepdim=True) / x.shape[-1])
    out_features = weight.shape[0]
    scale = torch.cat(
        [
            alpha_pre.expand(n),
            alpha_post.expand(n),
            alpha_res.expand(out_features - 2 * n),
        ],
        dim=0,
    ).float()
    linear = proj.float() * scale.unsqueeze(0) / (r + eps) + bias.float().unsqueeze(0)
    y = torch.cat(
        [
            linear[:, :n].sigmoid(),
            linear[:, n : 2 * n].sigmoid() * 2,
            linear[:, 2 * n :],
        ],
        dim=-1,
    )
    return y.to(x.dtype), r.to(x.dtype)

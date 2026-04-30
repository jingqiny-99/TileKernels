import torch
from tilelang.autotuner import set_autotune_inputs

from tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_bwd, _mhc_sinkhorn_fwd


class _SinkhornNormalize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: '_SinkhornNormalize',
        x: torch.Tensor,
        repeat: int,
        eps: float,
    ) -> torch.Tensor:
        hidden_size = x.shape[1]
        output = torch.empty_like(x)
        with set_autotune_inputs(x, output):
            fwd_kernel = _mhc_sinkhorn_fwd(hidden_size, token_block_size=1, repeat=repeat, eps=eps)
        ctx.save_for_backward(x)
        ctx.repeat = repeat
        ctx.eps = eps
        fwd_kernel(x, output)
        return output

    @staticmethod
    def backward(ctx: '_SinkhornNormalize', grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        x = ctx.saved_tensors[0]
        grad_input = torch.empty_like(x)
        hidden_size = x.shape[1]
        with set_autotune_inputs(grad_output, x, grad_input):
            bwd_kernel = _mhc_sinkhorn_bwd(hidden_size, token_block_size=32, repeat=ctx.repeat, eps=ctx.eps)
        bwd_kernel(grad_output, x, grad_input)
        return grad_input, None, None


def sinkhorn_normalize(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    return _SinkhornNormalize.apply(x.contiguous().view(-1, *x.shape[-2:]), repeat, eps).view_as(x)

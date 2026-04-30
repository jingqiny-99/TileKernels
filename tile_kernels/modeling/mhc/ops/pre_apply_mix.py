import torch
from tilelang.autotuner import set_autotune_inputs

from tile_kernels.mhc.pre_apply_mix_kernel import _mhc_pre_apply_mix_bwd, _mhc_pre_apply_mix_fwd


_BWD_AUTOTUNE_KEYS: set[tuple[int, int, int, str]] = set()


class MHCPreApplyMix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: 'MHCPreApplyMix',
        x: torch.Tensor,
        mix: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, mix)
        h = x.shape[-1]
        mhc = mix.shape[-2]
        assert mix.shape[-1] == 1
        if out is None:
            out = torch.empty(*x.shape[:-2], h, dtype=torch.bfloat16, device=x.device)
        with set_autotune_inputs(x.view(-1, mhc, h), mix.view(-1, mhc), out.view(-1, h)):
            fwd_kernel = _mhc_pre_apply_mix_fwd(mhc, h)
        fwd_kernel(x.view(-1, mhc, h), mix.view(-1, mhc), out.view(-1, h))
        return out

    @staticmethod
    def backward(ctx: 'MHCPreApplyMix', o_grad: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor, None]:
        x, mix = ctx.saved_tensors
        h = x.shape[-1]
        mhc = mix.shape[-2]
        tune_key = (mhc, h, o_grad.numel(), str(o_grad.device))
        if hasattr(x.untyped_storage(), 'grad_from_mhc_post'):
            x_grad = x.untyped_storage().grad_from_mhc_post
            if tune_key not in _BWD_AUTOTUNE_KEYS:
                tune_x_grad = x_grad.clone()
                with set_autotune_inputs(o_grad.view(-1, h), x.view(-1, mhc, h), mix.view(-1, mhc), tune_x_grad.view(-1, mhc, h)):
                    bwd_kernel = _mhc_pre_apply_mix_bwd(mhc, h)
                _BWD_AUTOTUNE_KEYS.add(tune_key)
            else:
                bwd_kernel = _mhc_pre_apply_mix_bwd(mhc, h)
            mix_grad = bwd_kernel(
                o_grad.view(-1, h),
                x.view(-1, mhc, h),
                mix.view(-1, mhc),
                x_grad.view(-1, mhc, h),
            )
            x_grad = None
        else:
            x_grad = torch.zeros_like(x)
            if tune_key not in _BWD_AUTOTUNE_KEYS:
                tune_x_grad = x_grad.clone()
                with set_autotune_inputs(o_grad.view(-1, h), x.view(-1, mhc, h), mix.view(-1, mhc), tune_x_grad.view(-1, mhc, h)):
                    bwd_kernel = _mhc_pre_apply_mix_bwd(mhc, h)
                _BWD_AUTOTUNE_KEYS.add(tune_key)
            else:
                bwd_kernel = _mhc_pre_apply_mix_bwd(mhc, h)
            mix_grad = bwd_kernel(
                o_grad.view(-1, h),
                x.view(-1, mhc, h),
                mix.view(-1, mhc),
                x_grad.view(-1, mhc, h),
            )
        return x_grad, mix_grad.view_as(mix), None


def mhc_pre_apply_mix(
    x: torch.Tensor,
    mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return MHCPreApplyMix.apply(x, mix, out)

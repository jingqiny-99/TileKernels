import torch
from tilelang.autotuner import set_autotune_inputs

from tile_kernels.mhc.norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul, round_to_tf32
from tile_kernels.mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)
    assert mhc_hidden_size % n_splits == 0

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device)

    gemm_out_mul = torch.empty(num_tokens, n_splits, mhc_mult3, dtype=torch.float32, device=residual.device)
    gemm_out_sqrsum = torch.empty(num_tokens, n_splits, dtype=torch.float32, device=residual.device)

    fn = round_to_tf32(fn)

    split_group_size = mhc_hidden_size // n_splits
    with set_autotune_inputs(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
    ):
        fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, n_splits, split_group_size)
    fwd_mul_kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
    )
    # END of TileLang implementation of pre-norm-fn forward matmul

    with set_autotune_inputs(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    ):
        pre_big_fuse_kernel = _mhc_pre_big_fuse(
            hidden_size,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            n_splits=n_splits,
            mhc_mult=mhc_mult,
        )
    pre_big_fuse_kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input

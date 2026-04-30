import pytest
import torch
from tile_kernels.modeling.mhc.ops.multilayer_recompute import mhc_multilayer_recompute
from tile_kernels.modeling.mhc.ops.post import mhc_post
from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix


_CORRECTNESS_CASES = [
    (1, 1, 2560),
    (3, 2, 2560),
    (3, 3, 2560),
    (10, 9, 2560),
    (10, 10, 2560),
    (10, 9, 4096),
    (10, 10, 4096),
    (10, 9, 7168),
    (10, 10, 7168),
    (10, 9, 8192),
    (10, 10, 8192),
]

def generate_multilayer_recompute_test_data(
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    initial_residual = torch.randn(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16)
    pre_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_layers)]
    layer_output_list = [torch.randn(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
    post_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_post)]
    comb_mix_list = [torch.randn(bs, seq, mhc_mult, mhc_mult, device='cuda', dtype=torch.float32) for _ in range(num_post)]
    layer_input_list = [torch.empty(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_layers)]
    residual_list = [torch.empty(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
    return initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list


def _mhc_multilayer_recompute_ref(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layer_input_refs: list[torch.Tensor] = []
    residual_refs: list[torch.Tensor] = []
    residual = initial_residual
    for i in range(len(pre_mix_list)):
        layer_input = mhc_pre_apply_mix(residual, pre_mix_list[i])
        layer_input_refs.append(layer_input)
        if i < len(layer_output_list):
            residual = mhc_post(layer_output_list[i], residual, post_mix_list[i], comb_mix_list[i])
            residual_refs.append(residual)
    return layer_input_refs, residual_refs


@pytest.mark.parametrize('num_layers,num_post,hidden', _CORRECTNESS_CASES)
def test_mhc_multilayer_recompute_correctness(num_layers: int, num_post: int, hidden: int) -> None:
    torch.manual_seed(0)
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        generate_multilayer_recompute_test_data(1, 8192, 4, hidden, num_layers, num_post)
    )
    layer_input_ref, residual_ref = _mhc_multilayer_recompute_ref(initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list)
    mhc_multilayer_recompute(initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list)

    for i in range(num_layers):
        assert torch.equal(layer_input_list[i], layer_input_ref[i]), (
            f'layer_input[{i}] mismatch! max diff = {(layer_input_list[i].float() - layer_input_ref[i].float()).abs().max().item()}'
        )
    for i in range(num_post):
        assert torch.equal(residual_list[i], residual_ref[i]), (
            f'residual[{i}] mismatch! max diff = {(residual_list[i].float() - residual_ref[i].float()).abs().max().item()}'
        )

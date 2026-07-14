import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op, is_310p

enable_custom_op()

DTYPE = torch.float16 if is_310p() else torch.bfloat16
ATOL = 3e-2 if DTYPE == torch.float16 else 2e-2
RTOL = 3e-2 if DTYPE == torch.float16 else 2e-2


@pytest.mark.parametrize("num_tokens", [1, 17, 129])
@pytest.mark.parametrize("num_q_heads, num_kv_heads", [(8, 2), (32, 8)])
@pytest.mark.parametrize("is_interleaved", [False, True])
@torch.inference_mode()
def test_ascendc_split_qkv_rmsnorm_mrope_matches_triton(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_interleaved: bool,
):
    torch.set_default_device("npu")
    init_device_properties_triton()
    torch.manual_seed(0)

    head_size = 128
    rope_dim = 128
    mrope_section = [16, 24, 24]
    max_positions = 4096
    eps = 1e-6
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    qkv = torch.randn(
        num_tokens,
        q_size + 2 * kv_size,
        dtype=DTYPE,
        device="npu",
    )
    q_weight = torch.randn(head_size, dtype=DTYPE, device="npu")
    k_weight = torch.randn(head_size, dtype=DTYPE, device="npu")
    cos_sin_cache = torch.randn(
        max_positions,
        rope_dim,
        dtype=DTYPE,
        device="npu",
    )
    positions = torch.randint(
        0,
        max_positions,
        (3, num_tokens),
        dtype=torch.int64,
        device="npu",
    )

    cos_sin = cos_sin_cache[positions]
    triton_q, triton_k, triton_v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin=cos_sin,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=eps,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rope_dim=rope_dim,
    )

    ascendc_q, ascendc_k, ascendc_v = torch.ops._C_ascend.npu_split_qkv_rmsnorm_mrope(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
    )

    torch.testing.assert_close(ascendc_q, triton_q, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ascendc_k, triton_k, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ascendc_v, triton_v, atol=0, rtol=0)

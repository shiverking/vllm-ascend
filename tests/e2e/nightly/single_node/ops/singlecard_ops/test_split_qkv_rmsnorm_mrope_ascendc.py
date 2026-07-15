import pytest
import torch

from vllm_ascend.utils import enable_custom_op


def prepare_cos_sin(cache, positions, sections, interleaved):
    tokens = positions.shape[1]
    half = sum(sections)
    offsets = torch.arange(half, device=cache.device)
    if interleaved:
        h_mask = (offsets % 3 == 1) & (offsets <= 3 * sections[1])
        w_mask = (offsets % 3 == 2) & (offsets <= 3 * sections[2])
        axes = torch.where(h_mask, 1, torch.where(w_mask, 2, 0))
    else:
        axes = torch.where(offsets < sections[0], 0, torch.where(offsets < sum(sections[:2]), 1, 2))
    gathered = cache[positions].float().permute(1, 2, 0)
    index = axes.view(1, half, 1).expand(tokens, half, 1)
    cos = gathered[:, :half].gather(2, index).squeeze(2)
    sin = gathered[:, half:].gather(2, index).squeeze(2)
    return torch.cat((cos, cos), -1).contiguous(), torch.cat((sin, sin), -1).contiguous()


def reference(
    qkv, q_weight, k_weight, cos, sin,
    num_q_heads, num_kv_heads, head_size, eps,
):
    tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(tokens, num_q_heads, head_size).float()
    k = k.view(tokens, num_kv_heads, head_size).float()
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps) * q_weight.float()
    k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + eps) * k_weight.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    half = cos.shape[-1] // 2

    def rope(x):
        rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
        return (x * cos + rotated * sin).to(qkv.dtype)

    return rope(q).view(tokens, q_size), rope(k).view(tokens, kv_size), v


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("interleaved", [False, True])
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope_ascendc(tokens, interleaved):
    assert enable_custom_op()
    dtype = torch.float16
    num_q_heads, num_kv_heads, head_size = 8, 2, 128
    sections = [16, 24, 24]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    torch.manual_seed(0)
    qkv = torch.randn(tokens, q_size + 2 * kv_size, device="npu", dtype=dtype)
    q_weight = torch.randn(head_size, device="npu", dtype=dtype)
    k_weight = torch.randn(head_size, device="npu", dtype=dtype)
    cache = torch.randn(4096, head_size, device="npu", dtype=dtype)
    positions_storage = torch.randint(
        0, 4096, (3, tokens + 7), device="npu", dtype=torch.int64
    )
    positions = positions_storage[:, :tokens]
    assert positions.stride(1) == 1 and not positions.is_contiguous()

    cos, sin = prepare_cos_sin(cache, positions, sections, interleaved)
    expected = reference(
        qkv, q_weight, k_weight, cos, sin,
        num_q_heads, num_kv_heads, head_size, 1e-6,
    )
    actual = torch.ops._C_ascend.npu_split_qkv_rmsnorm_mrope(
        qkv, q_weight, k_weight, cos, sin, num_q_heads,
        num_kv_heads, head_size, 1e-6, head_size,
    )
    tolerance = 3e-2
    torch.testing.assert_close(actual[0], expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual[1], expected[1], atol=tolerance, rtol=tolerance)
    assert len(actual) == 2
    actual_v = qkv.narrow(-1, q_size + kv_size, kv_size)
    torch.testing.assert_close(actual_v, expected[2], atol=0, rtol=0)

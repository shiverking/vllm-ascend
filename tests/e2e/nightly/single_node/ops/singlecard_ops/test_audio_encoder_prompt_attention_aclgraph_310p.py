import pytest
import torch
import torch_npu

from vllm_ascend._310p.audio_encoder_acl_graph import (
    build_padded_attention_mask,
)
from vllm_ascend.utils import is_310p as is_310p_hw


GRAPH_SIZE = 128
NUM_HEADS = 4
HEAD_SIZE = 128
SCALE = HEAD_SIZE**-0.5


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(
        query.float(),
        key.float().transpose(-1, -2),
    ) * SCALE
    scores.masked_fill_(mask[None, None], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value.float()).half()


def _prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return torch_npu.npu_prompt_flash_attention(
        query,
        key,
        value,
        atten_mask=mask,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        scale_value=SCALE,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BNSD",
        sparse_mode=0,
    )


@pytest.mark.skipif(
    not is_310p_hw(),
    reason="Tested separately on an Ascend 310P machine.",
)
@torch.inference_mode()
def test_audio_encoder_prompt_attention_aclgraph_updates_device_mask():
    torch.manual_seed(0)
    query = torch.randn(
        1,
        NUM_HEADS,
        GRAPH_SIZE,
        HEAD_SIZE,
        device="npu",
        dtype=torch.float16,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    mask_a = build_padded_attention_mask((100,))[0].npu()
    mask_b = build_padded_attention_mask((26, 44))[0].npu()

    eager_a = _prompt_attention(query, key, value, mask_a)
    eager_b = _prompt_attention(query, key, value, mask_b)
    reference_a = _reference_attention(query, key, value, mask_a)
    reference_b = _reference_attention(query, key, value, mask_b)
    torch.testing.assert_close(eager_a, reference_a, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(eager_b, reference_b, atol=3e-2, rtol=3e-2)

    static_query = query.clone()
    static_key = key.clone()
    static_value = value.clone()
    static_mask = mask_a.clone()
    for _ in range(2):
        _prompt_attention(
            static_query,
            static_key,
            static_value,
            static_mask,
        )
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = _prompt_attention(
            static_query,
            static_key,
            static_value,
            static_mask,
        )

    mask_address = static_mask.data_ptr()
    outputs = []
    for mask in (mask_a, mask_b, mask_a):
        static_mask.copy_(mask)
        assert static_mask.data_ptr() == mask_address
        graph.replay()
        torch.npu.synchronize()
        outputs.append(graph_output.clone())

    torch.testing.assert_close(outputs[0], eager_a, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(outputs[1], eager_b, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(outputs[2], eager_a, atol=3e-2, rtol=3e-2)
    assert not torch.allclose(outputs[0], outputs[1], atol=3e-2, rtol=3e-2)

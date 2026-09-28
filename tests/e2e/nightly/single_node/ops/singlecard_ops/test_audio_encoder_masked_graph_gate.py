"""910B gate for replaying audio attention with a mutable device mask.

Run this before enabling the dedicated Qwen3-ASR encoder-body graph pool.
All four graphs are kept alive so the test also detects graph-pool interactions.
"""

import math

import pytest
import torch
from vllm.platforms import current_platform

from vllm_ascend.utils import is_310p

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(is_310p(), reason="910B audio graph gate")


GRAPH_BUDGETS = (26, 104, 208, 416)
ATTENTION_ALIGNMENT = 128
NUM_HEADS = 4
HEAD_SIZE = 64


def _mask(topology: tuple[int, ...], budget: int) -> torch.Tensor:
    actual = sum(topology)
    assert 0 < actual <= budget
    aligned = math.ceil(budget / ATTENTION_ALIGNMENT) * ATTENTION_ALIGNMENT
    lengths = [*topology]
    if actual < budget:
        lengths.append(budget - actual)
    if aligned > budget:
        lengths.append(aligned - budget)
    mask = torch.ones((aligned, aligned), dtype=torch.bool)
    offset = 0
    for length in lengths:
        mask[offset : offset + length, offset : offset + length] = False
        offset += length
    return mask.contiguous()


def _attention(
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
        scale_value=HEAD_SIZE**-0.5,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BNSD",
        sparse_mode=0,
    )


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topology: tuple[int, ...],
) -> torch.Tensor:
    outputs = []
    offset = 0
    for length in topology:
        end = offset + length
        q = query[:, :, offset:end].float()
        k = key[:, :, offset:end].float()
        v = value[:, :, offset:end].float()
        scores = torch.matmul(q, k.transpose(-1, -2)) * HEAD_SIZE**-0.5
        outputs.append(torch.matmul(scores.softmax(-1), v))
        offset = end
    return torch.cat(outputs, dim=2).half()


@pytest.mark.parametrize("dtype", [torch.float16])
@torch.inference_mode()
def test_four_audio_masked_attention_graphs_replay_a_b_a(dtype: torch.dtype):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")

    topologies = {
        26: ((25,), (20, 4)),
        104: ((104,), (78, 25)),
        208: ((104, 104), (104, 78, 25)),
        416: ((104, 104, 104, 104), (104, 104, 104, 78, 25)),
    }
    torch.manual_seed(0)
    pool = current_platform.get_global_graph_pool()
    graphs = {}

    for budget in GRAPH_BUDGETS:
        aligned = math.ceil(budget / ATTENTION_ALIGNMENT) * ATTENTION_ALIGNMENT
        shape = (1, NUM_HEADS, aligned, HEAD_SIZE)
        query = torch.zeros(shape, device="npu", dtype=dtype)
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        mask = _mask(topologies[budget][0], budget).to("npu")

        for _ in range(2):
            _attention(query, key, value, mask)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=pool):
            output = _attention(query, key, value, mask)
        graphs[budget] = (graph, query, key, value, mask, output)
        torch.npu.synchronize()
        print(f"[AUDIO_GRAPH_GATE] captured budget={budget}", flush=True)

    for budget in GRAPH_BUDGETS:
        graph, query, key, value, mask, output = graphs[budget]
        for topology in (*topologies[budget], topologies[budget][0]):
            actual = sum(topology)
            print(
                f"[AUDIO_GRAPH_GATE] case budget={budget} "
                f"topology={list(topology)} padding={budget - actual}",
                flush=True,
            )
            inputs = [
                (torch.randn_like(tensor) * 0.1).contiguous()
                for tensor in (query, key, value)
            ]
            query.copy_(inputs[0])
            key.copy_(inputs[1])
            value.copy_(inputs[2])
            mask.copy_(_mask(topology, budget))

            # Verify the mask itself before testing graph replay. This catches
            # an operator/mask incompatibility independently of graph capture.
            padded_eager = _attention(query, key, value, mask)[:, :, :actual].clone()
            reference = _reference(query, key, value, topology)
            torch.testing.assert_close(
                padded_eager, reference, atol=3e-2, rtol=3e-2
            )
            print(
                f"[AUDIO_GRAPH_GATE] eager passed budget={budget} "
                f"topology={list(topology)}",
                flush=True,
            )

            graph.replay()
            replayed = output[:, :, :actual].clone()
            torch.npu.synchronize()
            torch.testing.assert_close(
                replayed, padded_eager, atol=3e-2, rtol=3e-2
            )
            torch.testing.assert_close(
                replayed, reference, atol=3e-2, rtol=3e-2
            )
            print(
                f"[AUDIO_GRAPH_GATE] replay passed budget={budget} "
                f"topology={list(topology)} padding={budget - actual}",
                flush=True,
            )

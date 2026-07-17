# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu


GRAPH_TOKENS = 64
NUM_HEADS = 8
HEAD_SIZE = 128
TOLERANCE = 3e-2
TOPOLOGY_CASES = (
    ((50, 12), (50, 12, 2)),
    ((50, 5), (50, 5, 9)),
    ((49,), (49, 7, 8)),
)


def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(query)
    torch_npu._npu_flash_attention_unpad(
        query=query,
        key=key,
        value=value,
        seq_len=sequence_lengths,
        scale_value=HEAD_SIZE**-0.5,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        out=output,
    )
    return output


def _assert_prefix_matches_unpadded(
    graph_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    real_topology: tuple[int, ...],
    graph_topology: tuple[int, ...],
) -> None:
    actual_tokens = sum(real_topology)
    real_sequence_lengths = torch.tensor(
        real_topology,
        dtype=torch.int32,
        device="cpu",
    )
    padded_sequence_lengths = torch.tensor(
        graph_topology,
        dtype=torch.int32,
        device="cpu",
    )
    unpadded_output = _flash_attention(
        query[:actual_tokens],
        key[:actual_tokens],
        value[:actual_tokens],
        real_sequence_lengths,
    )
    padded_output = _flash_attention(
        query,
        key,
        value,
        padded_sequence_lengths,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        padded_output[:actual_tokens].cpu(),
        unpadded_output.cpu(),
        atol=TOLERANCE,
        rtol=TOLERANCE,
    )
    torch.testing.assert_close(
        graph_output[:actual_tokens].cpu(),
        unpadded_output.cpu(),
        atol=TOLERANCE,
        rtol=TOLERANCE,
    )


@torch.inference_mode()
def test_flash_attention_aclgraph_replays_dynamic_cpu_topology():
    """Gate padded encoder graphs on dynamic host topology replay support."""
    torch.manual_seed(0)
    query = torch.randn(
        GRAPH_TOKENS,
        NUM_HEADS,
        HEAD_SIZE,
        dtype=torch.float16,
        device="npu",
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    static_sequence_lengths = torch.tensor(
        TOPOLOGY_CASES[0][1],
        dtype=torch.int32,
        device="cpu",
    ).contiguous()
    host_address = static_sequence_lengths.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        graph_output = _flash_attention(
            query,
            key,
            value,
            static_sequence_lengths,
        )

    replay_order = (
        TOPOLOGY_CASES[0],
        TOPOLOGY_CASES[1],
        TOPOLOGY_CASES[2],
        TOPOLOGY_CASES[0],
    )
    for real_topology, graph_topology in replay_order:
        static_sequence_lengths.copy_(
            torch.tensor(graph_topology, dtype=torch.int32, device="cpu")
        )
        assert static_sequence_lengths.data_ptr() == host_address
        graph.replay()
        _assert_prefix_matches_unpadded(
            graph_output,
            query,
            key,
            value,
            real_topology,
            graph_topology,
        )

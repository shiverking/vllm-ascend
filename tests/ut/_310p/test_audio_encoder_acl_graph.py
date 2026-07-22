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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend._310p.audio_encoder_acl_graph import (
    AUDIO_ENCODER_PROMPT_GRAPH_SIZE,
    AudioEncoderAclGraphPool,
    FixedAudioEncoderAclGraphRunner,
    build_padded_attention_mask,
)


def _make_encoder(enforce_eager: bool = False):
    return SimpleNamespace(enforce_eager=enforce_eager)


@pytest.mark.parametrize(
    ("topology", "dummy_tokens"),
    [
        ((100,), 28),
        ((26, 44), 58),
        ((104,), 24),
        ((26, 26, 26, 26), 24),
        ((128,), 0),
        ((1,), 127),
    ],
)
def test_build_padded_attention_mask(topology, dummy_tokens):
    mask, actual_dummy_tokens = build_padded_attention_mask(topology)

    assert mask.shape == (128, 128)
    assert mask.dtype == torch.bool
    assert mask.is_contiguous()
    assert actual_dummy_tokens == dummy_tokens

    graph_topology = topology + ((dummy_tokens,) if dummy_tokens else ())
    offset = 0
    for length in graph_topology:
        end = offset + length
        assert not mask[offset:end, offset:end].any()
        if offset:
            assert mask[offset:end, :offset].all()
            assert mask[:offset, offset:end].all()
        offset = end


def test_build_padded_attention_mask_supports_larger_graph():
    mask, dummy_tokens = build_padded_attention_mask((104, 52), 256)

    assert mask.shape == (256, 256)
    assert dummy_tokens == 100
    assert not mask[:104, :104].any()
    assert not mask[104:156, 104:156].any()
    assert not mask[156:, 156:].any()
    assert mask[:104, 104:].all()


@pytest.mark.parametrize("topology", [(), (0,), (-1,), (129,)])
def test_build_padded_attention_mask_rejects_invalid_topology(topology):
    with pytest.raises(ValueError):
        build_padded_attention_mask(topology)


def test_audio_encoder_aclgraph_accepts_aligned_graph_pool():
    eager_forward = lambda *args: args[1]
    sizes = (128, 256, 384, 512, 640, 768, 896, 1024)
    pool = AudioEncoderAclGraphPool(
        _make_encoder(),
        eager_forward,
        sizes,
    )

    assert pool.graph_sizes == sizes
    with pytest.raises(ValueError, match=r"multiples of 128"):
        AudioEncoderAclGraphPool(_make_encoder(), eager_forward, (104,))


@pytest.mark.parametrize(
    ("topology", "expected_graphs", "expected_eager"),
    [
        ((13,), [128], 0),
        ((26,), [128], 0),
        ((52,), [128], 0),
        ((104,), [128], 0),
        ((130,), [256], 0),
        ((104, 104, 104, 104, 104), [640], 0),
        ((104,) * 10, [1024, 128], 0),
        ((1200, 104), [128], 1200),
    ],
)
def test_audio_encoder_aclgraph_builds_nearest_sequence_plan(
    topology,
    expected_graphs,
    expected_eager,
):
    sizes = (128, 256, 384, 512, 640, 768, 896, 1024)
    pool = AudioEncoderAclGraphPool(
        _make_encoder(),
        lambda *args: args[1],
        sizes,
    )

    plan = pool._build_plan(topology)
    graph_sizes = [
        chunk.runner.num_tokens
        for chunk in plan
        if chunk.runner is not None
    ]
    eager_tokens = sum(
        chunk.actual_tokens for chunk in plan if chunk.runner is None
    )

    assert graph_sizes == expected_graphs
    assert eager_tokens == expected_eager
    assert [chunk.sequence_start for chunk in plan] == [
        0,
        *[chunk.sequence_end for chunk in plan[:-1]],
    ]


def test_audio_encoder_aclgraph_long_standard_topology_has_no_eager_tail():
    sizes = (128, 256, 384, 512, 640, 768, 896, 1024)
    pool = AudioEncoderAclGraphPool(
        _make_encoder(),
        lambda *args: args[1],
        sizes,
    )

    plan = pool._build_plan((104,) * 23)

    assert all(chunk.runner is not None for chunk in plan)
    assert sum(chunk.actual_tokens for chunk in plan) == 104 * 23


def test_audio_encoder_aclgraph_replays_a_b_a_with_stable_mask_address(
    capsys,
):
    def eager_forward(encoder, hidden_states, *args):
        return hidden_states + 1

    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        eager_forward,
        128,
    )
    graph = mock.Mock()
    capture_input = torch.zeros(128, 4, dtype=torch.float16)
    capture_cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)
    capture_sequence_lengths = torch.tensor([128], dtype=torch.int32)

    with (
        mock.patch.object(torch.npu, "synchronize"),
        mock.patch.object(torch.npu, "NPUGraph", return_value=graph),
        mock.patch.object(torch.npu, "graph", return_value=nullcontext()),
        mock.patch(
            "vllm_ascend._310p.audio_encoder_acl_graph.current_platform.get_global_graph_pool",
            return_value=None,
        ),
    ):
        assert runner.capture(
            capture_input,
            capture_cu_seqlens,
            None,
            capture_sequence_lengths,
        )

    topologies = ((100,), (26, 44), (100,))
    masks = []
    mask_addresses = []
    outputs = []
    with mock.patch.object(runner, "_eligibility_error", return_value=None):
        for request_id, topology in enumerate(topologies, start=1):
            actual_tokens = sum(topology)
            hidden_states = torch.full(
                (actual_tokens, 4),
                float(request_id),
                dtype=torch.float16,
            )
            output = runner.run(
                hidden_states,
                torch.tensor([0, actual_tokens], dtype=torch.int32),
                None,
                torch.tensor(topology, dtype=torch.int32),
                len(topology),
                request_id=request_id,
            )
            outputs.append(output)
            masks.append(runner.static_attention_mask.clone())
            mask_addresses.append(runner.static_attention_mask.data_ptr())

    assert graph.replay.call_count == 3
    assert len(set(mask_addresses)) == 1
    torch.testing.assert_close(masks[0], masks[2])
    assert not torch.equal(masks[0], masks[1])
    assert [output.shape[0] for output in outputs] == [100, 70, 100]
    assert all(output.data_ptr() != runner.static_output.data_ptr() for output in outputs)
    assert torch.count_nonzero(runner.static_input[100:]) == 0
    logs = capsys.readouterr().out
    assert logs.count("graph_hit=True") == 3
    assert "actual=70, padded=128, seq_lens=[26, 44], dummy=58" in logs


def test_audio_encoder_aclgraph_falls_back_for_token_overflow(capsys):
    eager_forward = mock.Mock(side_effect=lambda encoder, hidden_states, *args: hidden_states)
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        eager_forward,
        128,
    )
    runner.graph = mock.Mock()
    hidden_states = mock.Mock()
    hidden_states.dtype = torch.float16
    hidden_states.device = torch.device("npu")
    hidden_states.shape = (129, 4)
    hidden_states.is_contiguous.return_value = True
    sequence_lengths = torch.tensor([129], dtype=torch.int32)

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        output = runner.run(
            hidden_states,
            torch.tensor([0, 129], dtype=torch.int32),
            None,
            sequence_lengths,
            1,
            request_id=7,
        )

    assert output is hidden_states
    eager_forward.assert_called_once()
    assert (
        "request=7, graph_hit=False, reason=token_overflow, "
        "actual=129, limit=128"
    ) in capsys.readouterr().out


def test_audio_encoder_aclgraph_accepts_multiple_audio_sequences():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        128,
    )
    runner.graph = mock.Mock()
    hidden_states = mock.Mock()
    hidden_states.dtype = torch.float16
    hidden_states.device = torch.device("npu")
    hidden_states.shape = (70, 4)
    hidden_states.is_contiguous.return_value = True

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        error = runner._eligibility_error(
            hidden_states,
            torch.tensor([26, 44], dtype=torch.int32),
        )

    assert error is None

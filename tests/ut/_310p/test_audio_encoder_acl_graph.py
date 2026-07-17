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

import torch

from vllm_ascend._310p.audio_encoder_acl_graph import (
    FixedAudioEncoderAclGraphRunner,
)


def _make_encoder(enforce_eager: bool = False):
    return SimpleNamespace(enforce_eager=enforce_eager)


def _make_hidden_states(tokens: int):
    hidden_states = mock.Mock()
    hidden_states.dtype = torch.float16
    hidden_states.device = torch.device("npu")
    hidden_states.shape = (tokens, 8)
    hidden_states.is_contiguous.return_value = True
    return hidden_states


def test_audio_encoder_aclgraph_builds_padding_topology():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        64,
    )

    assert runner._build_graph_topology((50, 12), 62) == (50, 12, 2)
    assert runner._build_graph_topology((60,), 60) == (60, 2, 2)
    assert runner._build_graph_topology((63,), 63) is None


def test_audio_encoder_aclgraph_accepts_padding_and_ignores_num_audios():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        64,
    )

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        topology = runner._check_eligibility(
            _make_hidden_states(62),
            torch.tensor([50, 12], dtype=torch.int32),
            5,
        )

    assert topology == (50, 12, 2)


def test_audio_encoder_aclgraph_rejects_excessive_padding():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        64,
    )

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        topology = runner._check_eligibility(
            _make_hidden_states(55),
            torch.tensor([50, 5], dtype=torch.int32),
            1,
        )

    assert topology is None
    assert "fallback:padding_ratio" in runner._log_keys


def test_audio_encoder_aclgraph_rejects_exact_size():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        64,
    )

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        topology = runner._check_eligibility(
            _make_hidden_states(64),
            torch.tensor([50, 14], dtype=torch.int32),
            1,
        )

    assert topology is None
    assert "fallback:exact_graph_size" in runner._log_keys


def test_audio_encoder_aclgraph_rejects_too_many_sequences():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        64,
    )

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        topology = runner._check_eligibility(
            _make_hidden_states(62),
            torch.tensor([30, 20, 12], dtype=torch.int32),
            1,
        )

    assert topology is None
    assert "fallback:too_many_sequences" in runner._log_keys


def test_audio_encoder_aclgraph_captures_once_then_replays_and_clones():
    calls = []

    def eager_forward(encoder, hidden_states, *args):
        calls.append(hidden_states)
        return hidden_states + 1

    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        eager_forward,
        64,
    )
    graph = mock.Mock()
    hidden_states = torch.randn(62, 8, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 50, 62], dtype=torch.int32)
    sequence_lengths = torch.tensor([50, 12], dtype=torch.int32)

    with (
        mock.patch.object(
            runner,
            "_check_eligibility",
            return_value=(50, 12, 2),
        ),
        mock.patch.object(torch.npu, "synchronize"),
        mock.patch.object(torch.npu, "NPUGraph", return_value=graph),
        mock.patch.object(torch.npu, "graph", return_value=nullcontext()),
        mock.patch(
            "vllm_ascend._310p.audio_encoder_acl_graph."
            "current_platform.get_global_graph_pool",
            return_value=None,
        ),
    ):
        first = runner.run(
            hidden_states,
            cu_seqlens,
            None,
            sequence_lengths,
            5,
        )
        assert runner.static_input is not None
        runner.static_input[62:].fill_(7)
        host_address = runner.static_sequence_lengths.data_ptr()
        second = runner.run(
            hidden_states,
            cu_seqlens,
            None,
            sequence_lengths,
            5,
        )

    assert len(calls) == 3
    graph.replay.assert_called_once_with()
    assert first.shape[0] == 62
    assert second.shape[0] == 62
    assert first.data_ptr() != runner.static_output.data_ptr()
    assert second.data_ptr() != runner.static_output.data_ptr()
    assert torch.count_nonzero(runner.static_input[62:]) == 0
    assert runner.static_sequence_lengths.tolist() == [50, 12, 2]
    assert runner.static_sequence_lengths.data_ptr() == host_address

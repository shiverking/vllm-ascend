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
    return SimpleNamespace(
        n_window=100,
        n_window_infer=400,
        enforce_eager=enforce_eager,
    )


def test_audio_encoder_aclgraph_builds_fixed_topology():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        125,
    )

    assert runner.expected_sequence_lengths == (50, 50, 25)


def test_audio_encoder_aclgraph_rejects_topology_mismatch():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        125,
    )
    hidden_states = mock.Mock()
    hidden_states.dtype = torch.float16
    hidden_states.device = torch.device("npu")
    hidden_states.shape = (125, 8)
    hidden_states.is_contiguous.return_value = True

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        eligible = runner._check_eligibility(
            hidden_states,
            torch.tensor([50, 50, 24, 1], dtype=torch.int32),
            1,
        )

    assert not eligible


def test_audio_encoder_aclgraph_rejects_token_mismatch():
    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        lambda *args: args[1],
        125,
    )
    hidden_states = mock.Mock()
    hidden_states.dtype = torch.float16
    hidden_states.device = torch.device("npu")
    hidden_states.shape = (124, 8)
    hidden_states.is_contiguous.return_value = True

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        eligible = runner._check_eligibility(
            hidden_states,
            torch.tensor([50, 50, 24], dtype=torch.int32),
            1,
        )

    assert not eligible


def test_audio_encoder_aclgraph_captures_once_then_replays_and_clones():
    calls = []

    def eager_forward(encoder, hidden_states, *args):
        calls.append(hidden_states)
        return hidden_states + 1

    runner = FixedAudioEncoderAclGraphRunner(
        _make_encoder(),
        eager_forward,
        125,
    )
    graph = mock.Mock()
    hidden_states = torch.randn(125, 8, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 50, 100, 125], dtype=torch.int32)
    sequence_lengths = torch.tensor([50, 50, 25], dtype=torch.int32)

    with (
        mock.patch.object(runner, "_check_eligibility", return_value=True),
        mock.patch.object(torch.npu, "synchronize"),
        mock.patch.object(torch.npu, "NPUGraph", return_value=graph),
        mock.patch.object(torch.npu, "graph", return_value=nullcontext()),
        mock.patch(
            "vllm_ascend._310p.audio_encoder_acl_graph.current_platform.get_global_graph_pool",
            return_value=None,
        ),
    ):
        first = runner.run(
            hidden_states,
            cu_seqlens,
            None,
            sequence_lengths,
            1,
        )
        second = runner.run(
            hidden_states,
            cu_seqlens,
            None,
            sequence_lengths,
            1,
        )

    assert len(calls) == 3
    graph.replay.assert_called_once_with()
    assert first.data_ptr() != runner.static_output.data_ptr()
    assert second.data_ptr() != runner.static_output.data_ptr()

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
    AudioEncoderAclGraphPool,
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


def test_audio_encoder_aclgraph_captures_once_then_replays_and_clones(capsys):
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
    output = capsys.readouterr().out
    assert (
        "[310P_AUDIO_GRAPH] request=1, graph_hit=False, action=capture"
        in output
    )


def _make_encoder_with_104_token_window():
    return SimpleNamespace(
        n_window=50,
        n_window_infer=800,
        enforce_eager=False,
    )
    assert (
        "[310P_AUDIO_GRAPH] request=2, graph_hit=True, action=replay"
        in output
    )


def test_audio_encoder_aclgraph_logs_every_fallback(capsys):
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
    cu_seqlens = torch.tensor([0, 50, 100, 124], dtype=torch.int32)
    sequence_lengths = torch.tensor([50, 50, 24], dtype=torch.int32)

    with mock.patch(
        "vllm_ascend._310p.audio_encoder_acl_graph.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        for _ in range(2):
            runner.run(
                hidden_states,
                cu_seqlens,
                None,
                sequence_lengths,
                1,
            )

    output = capsys.readouterr().out
    assert output.count("graph_hit=False, action=fallback") == 2
    assert "request=1, graph_hit=False, action=fallback" in output
    assert "request=2, graph_hit=False, action=fallback" in output
    assert output.count("reason=token_mismatch") == 2


def test_audio_encoder_aclgraph_pool_uses_largest_sequence_aligned_graph():
    pool = AudioEncoderAclGraphPool(
        _make_encoder_with_104_token_window(),
        lambda *args: args[1],
        (104, 312, 520),
    )

    chunks, tail_sequence_start, tail_token_start = pool._build_plan((104,) * 15)

    assert [chunk.runner.num_tokens for chunk in chunks] == [520, 520, 520]
    assert tail_sequence_start == 15
    assert tail_token_start == 1560


def test_audio_encoder_aclgraph_pool_leaves_unmatched_tail_for_eager():
    pool = AudioEncoderAclGraphPool(
        _make_encoder_with_104_token_window(),
        lambda *args: args[1],
        (104, 312, 520),
    )
    topology = (104,) * 14 + (44,)

    chunks, tail_sequence_start, tail_token_start = pool._build_plan(topology)

    assert [chunk.runner.num_tokens for chunk in chunks] == [
        520,
        520,
        312,
        104,
    ]
    assert tail_sequence_start == 14
    assert tail_token_start == 1456


def test_audio_encoder_aclgraph_pool_runs_graph_chunks_and_tail_once(capsys):
    def eager_forward(encoder, hidden_states, *args):
        return hidden_states + 10

    pool = AudioEncoderAclGraphPool(
        _make_encoder(),
        eager_forward,
        (100,),
    )
    runner = pool.runners[100]
    hidden_states = torch.arange(124 * 2, dtype=torch.float16).view(124, 2)
    cu_seqlens = torch.tensor([0, 50, 100, 124], dtype=torch.int32)
    sequence_lengths = torch.tensor([50, 50, 24], dtype=torch.int32)

    def run_graph_chunk(chunk_hidden_states, *args, **kwargs):
        runner.last_action = "replay"
        return chunk_hidden_states + 1

    with (
        mock.patch.object(pool, "_check_common_eligibility", return_value=None),
        mock.patch.object(runner, "run", side_effect=run_graph_chunk),
    ):
        output = pool.run(
            hidden_states,
            cu_seqlens,
            None,
            sequence_lengths,
            1,
        )

    torch.testing.assert_close(output[:100], hidden_states[:100] + 1)
    torch.testing.assert_close(output[100:], hidden_states[100:] + 10)
    logs = capsys.readouterr().out
    assert "graph_chunks=[100], eager_tail_tokens=24" in logs
    assert "action=eager_tail, tokens=24, seq_lens=[24]" in logs
    assert "replay_hits=1" in logs
    assert "eager_tail_tokens=24" in logs

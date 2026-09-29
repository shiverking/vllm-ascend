from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.worker.encoder_acl_graph import (
    EncoderAclGraphManager,
    build_audio_attention_mask,
)
from vllm_ascend.worker.qwen3_audio_encoder_graph import (
    FixedAudioEncoderGraph,
    Qwen3AudioEncoderGraphPool,
    plan_audio_graph_chunks,
)


BUDGETS = (26, 104, 208, 416)


@pytest.mark.parametrize(
    "topology, expected_budgets, expected_padding",
    [
        ((25,), [26], 1),
        ((26,), [26], 0),
        ((27,), [104], 77),
        ((52,), [104], 52),
        ((78,), [104], 26),
        ((104,), [104], 0),
        ((104, 1), [104, 26], 25),
        ((104, 24), [104, 26], 2),
        ((104, 104), [208], 0),
        ((104, 104, 48), [208, 104], 56),
        ((104, 104, 104, 72), [416], 32),
        ((104, 104, 104, 104), [416], 0),
        ((104, 104, 104, 104, 96), [416, 104], 8),
    ],
)
def test_audio_graph_plan_covers_complete_sequences(
    topology, expected_budgets, expected_padding
):
    plan = plan_audio_graph_chunks(topology, BUDGETS)
    assert [chunk.budget for chunk in plan] == expected_budgets
    assert sum(chunk.actual_tokens for chunk in plan) == sum(topology)
    assert sum(
        chunk.budget - chunk.actual_tokens for chunk in plan
    ) == expected_padding
    offsets = [0]
    for length in topology:
        offsets.append(offsets[-1] + length)
    for chunk in plan:
        assert chunk.token_start == offsets[chunk.sequence_start]
        assert chunk.token_end == offsets[chunk.sequence_end]


def test_audio_graph_plan_reuses_basis_graphs_for_long_audio():
    plan = plan_audio_graph_chunks((104,) * 15, BUDGETS)
    assert [chunk.budget for chunk in plan] == [416, 416, 416, 208, 104]
    assert all(chunk.budget is not None for chunk in plan)


def test_audio_graph_mask_isolates_real_and_dummy_sequences():
    mask = build_audio_attention_mask((104, 24), 208)
    assert mask.shape == (256, 256)
    assert not mask[:104, :104].any()
    assert not mask[104:128, 104:128].any()
    assert not mask[128:208, 128:208].any()
    assert mask[:104, 104:128].all()
    assert mask[104:128, 128:208].all()
    assert mask[:208, 208:].all()


@pytest.mark.parametrize("topology", [(), (0,), (-1,), (27, 0)])
def test_audio_graph_plan_rejects_invalid_topology(topology):
    with pytest.raises(ValueError, match="positive"):
        plan_audio_graph_chunks(topology, BUDGETS)


def test_audio_graph_mask_rejects_over_budget():
    with pytest.raises(ValueError, match="exceeds"):
        build_audio_attention_mask(torch.tensor([104, 25]), 104)


def test_audio_graph_capture_orders_smallest_first():
    model = MagicMock()
    model.audio_tower.get_attention_window_tokens.return_value = 104
    pool = Qwen3AudioEncoderGraphPool(
        model, torch.device("cpu"), (416, 26, 208, 104)
    )
    captured = []
    for budget, runner in pool.runners.items():
        runner.capture = lambda _pool, size=budget: captured.append(size)
    with (
        patch("vllm_ascend.worker.qwen3_audio_encoder_graph.torch.npu.Stream"),
        patch("vllm_ascend.worker.qwen3_audio_encoder_graph.torch.npu.Event"),
        patch(
            "vllm_ascend.worker.qwen3_audio_encoder_graph.torch.npu.stream",
            return_value=nullcontext(),
        ),
    ):
        pool.capture(graph_pool=object())
    assert captured == [26, 104, 208, 416]


def test_audio_graph_capture_stops_at_first_failure():
    model = MagicMock()
    model.audio_tower.get_attention_window_tokens.return_value = 104
    pool = Qwen3AudioEncoderGraphPool(model, torch.device("cpu"), BUDGETS)
    captured = []

    def capture(size):
        captured.append(size)
        if size == 104:
            raise RuntimeError("capture failure")

    for budget, runner in pool.runners.items():
        runner.capture = lambda _pool, size=budget: capture(size)
    with (
        patch("vllm_ascend.worker.qwen3_audio_encoder_graph.torch.npu.Stream"),
        patch("vllm_ascend.worker.qwen3_audio_encoder_graph.torch.npu.Event"),
        patch(
            "vllm_ascend.worker.qwen3_audio_encoder_graph.torch.npu.stream",
            return_value=nullcontext(),
        ),
        pytest.raises(RuntimeError, match="104 tokens"),
    ):
        pool.capture(graph_pool=object())
    assert captured == [26, 104]


def test_audio_graph_replay_clones_output_and_clears_padding():
    runner = FixedAudioEncoderGraph(MagicMock(), 26, torch.device("cpu"))
    runner.hidden_states = torch.full((26, 2), -1.0)
    runner.mask = torch.empty((128, 128), dtype=torch.bool)
    runner.host_mask = torch.empty_like(runner.mask)
    runner.output = torch.empty((26, 2))
    runner.graph = MagicMock()
    runner.graph.replay.side_effect = lambda: runner.output.copy_(
        runner.hidden_states
    )

    first = runner.replay(torch.ones((25, 2)), (25,))
    second = runner.replay(torch.full((20, 2), 2.0), (20,))

    assert torch.all(first == 1)
    assert torch.all(second == 2)
    assert torch.all(runner.hidden_states[20:] == 0)
    assert first.data_ptr() != runner.output.data_ptr()


@pytest.mark.parametrize(
    "graph_chunks,padding,eager_tokens,expected_hit",
    [
        ([104, 26], 2, 0, True),
        ([104], 0, 24, False),
        ([], 0, 128, False),
    ],
)
def test_audio_graph_call_log_exposes_capture_size_inputs(
    graph_chunks, padding, eager_tokens, expected_hit
):
    model = MagicMock()
    model.audio_tower.get_attention_window_tokens.return_value = 104
    hidden_states = MagicMock()
    hidden_states.dtype = torch.float16
    hidden_states.device = torch.device("npu")
    hidden_states.shape = (128, 8)
    hidden_states.is_contiguous.return_value = True
    model._prepare_audio_encoder_graph_inputs.return_value = (
        hidden_states, None, None, torch.tensor([104, 24], dtype=torch.int32)
    )
    model.get_encoder_cudagraph_item_specs.return_value = [
        SimpleNamespace(output_tokens=128)
    ]
    pool = Qwen3AudioEncoderGraphPool(model, torch.device("cpu"), BUDGETS)

    with (
        patch.object(pool, "_ordered_stream", return_value=nullcontext()),
        patch.object(
            pool,
            "_execute_plan",
            return_value=(torch.zeros(128, 8), graph_chunks, padding, eager_tokens),
        ),
        patch("vllm_ascend.worker.qwen3_audio_encoder_graph.logger.info") as log_info,
    ):
        pool.execute({})

    message = log_info.call_args.args[0] % log_info.call_args.args[1:]
    assert "items=1 item_tokens=[128] actual=128 seq_lens=[104, 24]" in message
    assert f"graphs={graph_chunks} replays={len(graph_chunks)}" in message
    assert f"padding={padding} eager={eager_tokens}" in message
    assert f"graph_used={bool(graph_chunks)} graph_hit={expected_hit}" in message


def test_qwen_audio_uses_dedicated_manager_not_generic_capture():
    manager = object.__new__(EncoderAclGraphManager)
    manager._is_qwen3_asr = True
    manager._qwen_audio_pool = MagicMock()
    manager._qwen_audio_pool.execute.return_value = [torch.ones(2, 3)]
    graph_pool = object()

    manager.capture(graph_pool)
    result = manager.execute({"audio": object()})

    manager._qwen_audio_pool.capture.assert_called_once_with(graph_pool)
    manager._qwen_audio_pool.execute.assert_called_once()
    assert len(result) == 1
    assert manager.supports_modality("audio")
    assert not manager.supports_modality("image")

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.qwen3_audio_encoder_graph import (
    Qwen3AudioEncoderGraphPool,
    plan_audio_graph_chunks,
)


def _model():
    return SimpleNamespace(
        audio_tower=SimpleNamespace(get_attention_window_tokens=lambda: 104)
    )


def test_audio_graph_pool_accepts_custom_budgets():
    pool = Qwen3AudioEncoderGraphPool(
        _model(), torch.device("cpu"), [416, 52, 26, 52, 104, 208]
    )
    assert pool.budgets == (26, 52, 104, 208, 416)


@pytest.mark.parametrize("budgets", [[], [0, 104], [-1, 104]])
def test_audio_graph_pool_rejects_invalid_budgets(budgets):
    with pytest.raises(ValueError, match="must be positive"):
        Qwen3AudioEncoderGraphPool(_model(), torch.device("cpu"), budgets)


def test_audio_graph_pool_reuses_budgets_for_large_batch():
    plan = plan_audio_graph_chunks([104] * 80, [26, 104, 208, 416])
    assert len(plan) == 20
    assert all(chunk.budget == 416 for chunk in plan)

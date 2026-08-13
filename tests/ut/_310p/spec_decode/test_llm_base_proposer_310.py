# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import (
    AscendSpecDecodeBaseProposer310,
)
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def test_run_merged_draft_restores_rope_flag() -> None:
    observed_states = []

    def mock_original(*_args, **_kwargs):
        observed_states.append(AscendRotaryEmbedding310._is_drafting_update_enabled)
        return torch.zeros(4, dtype=torch.long)

    with patch(
        "vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft",
        mock_original,
    ):
        proposer = object.__new__(AscendSpecDecodeBaseProposer310)
        proposer._run_merged_draft(
            num_input_tokens=4,
            batch_size=1,
            token_indices_to_sample=torch.tensor([3]),
            target_positions=torch.arange(4),
            inputs_embeds=torch.zeros(4, 8),
            multi_steps_attn_metadata=None,
            num_tokens=4,
        )

    assert observed_states == [True]
    assert not AscendRotaryEmbedding310._is_drafting_update_enabled


def test_run_merged_draft_restores_rope_flag_after_failure() -> None:
    def mock_original(*_args, **_kwargs):
        raise RuntimeError("draft failed")

    with patch(
        "vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft",
        mock_original,
    ):
        proposer = object.__new__(AscendSpecDecodeBaseProposer310)
        with pytest.raises(RuntimeError, match="draft failed"):
            proposer._run_merged_draft(
                num_input_tokens=1,
                batch_size=1,
                token_indices_to_sample=torch.tensor([0]),
                target_positions=torch.tensor([0]),
                inputs_embeds=torch.zeros(1, 8),
                multi_steps_attn_metadata=None,
                num_tokens=1,
            )

    assert not AscendRotaryEmbedding310._is_drafting_update_enabled


def test_set_inputs_first_pass_preserves_tail_slot() -> None:
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.needs_extra_input_slots = False
    proposer.input_ids = torch.full((6,), 99, dtype=torch.long)
    proposer.hidden_states = torch.zeros(6, 8)
    proposer.runner = object()
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 3, 6]))

    with patch.object(AscendSpecDecodeBaseProposer, "_set_positions"):
        proposer.set_inputs_first_pass(
            target_token_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
            next_token_ids=torch.tensor([10, 20]),
            target_positions=torch.arange(6),
            target_hidden_states=torch.ones(6, 8),
            token_indices_to_sample=torch.tensor([1, 4]),
            cad=metadata,
            num_rejected_tokens_gpu=None,
        )

    torch.testing.assert_close(
        proposer.input_ids,
        torch.tensor([2, 10, 4, 5, 20, 99]),
    )

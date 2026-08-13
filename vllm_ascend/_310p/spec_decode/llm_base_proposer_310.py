#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

_original_run_merged_draft = AscendSpecDecodeBaseProposer._run_merged_draft


class AscendSpecDecodeBaseProposer310(AscendSpecDecodeBaseProposer):
    """310P proposer overrides for NPU-specific spec-decode workarounds."""

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            return _original_run_merged_draft(
                self,
                num_input_tokens,
                batch_size,
                token_indices_to_sample,
                target_positions,
                inputs_embeds,
                multi_steps_attn_metadata,
                num_tokens,
                is_prefill,
            )
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            tail_token_id = self.input_ids[num_tokens - 1].clone()
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            self.input_ids[num_tokens - 1] = tail_token_id
            self.input_ids[token_indices_to_sample] = next_token_ids

            assert self.runner is not None
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states.view(num_tokens, -1)
            return num_tokens, token_indices_to_sample, cad, (None, None)

        return super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )

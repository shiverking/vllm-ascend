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

from __future__ import annotations

import torch
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

from vllm_ascend._310p.audio_encoder_acl_graph import (
    AudioEncoderAclGraphPool,
)
from vllm_ascend.ascend_config import get_ascend_config


_original_forward_encoder_body = Qwen3OmniMoeAudioEncoder._forward_encoder_body


def _forward_encoder_body_with_aclgraph(
    self: Qwen3OmniMoeAudioEncoder,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor | None,
    sequence_lengths: torch.Tensor,
    num_audios: int = 1,
) -> torch.Tensor:
    if (
        sequence_lengths.device.type == "cpu"
        and sequence_lengths.dtype == torch.int32
        and not getattr(self, "_ascend_sequence_lengths_reuse_logged", False)
    ):
        print(
            "[AUDIO_ENCODER_D2H] reused precomputed CPU sequence lengths",
            flush=True,
        )
        self._ascend_sequence_lengths_reuse_logged = True

    graph_sizes = get_ascend_config().audio_encoder_aclgraph_sizes
    if not graph_sizes:
        return _original_forward_encoder_body(
            self,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
            num_audios,
        )

    pool = getattr(self, "_ascend_audio_aclgraph_pool", None)
    if pool is None or pool.graph_sizes != graph_sizes:
        pool = AudioEncoderAclGraphPool(
            self,
            _original_forward_encoder_body,
            graph_sizes,
        )
        self._ascend_audio_aclgraph_pool = pool
    return pool.run(
        hidden_states,
        cu_seqlens,
        max_seqlen,
        sequence_lengths,
        num_audios,
    )


Qwen3OmniMoeAudioEncoder._forward_encoder_body = _forward_encoder_body_with_aclgraph

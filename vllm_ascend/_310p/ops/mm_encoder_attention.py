#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import einops
import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore

from vllm_ascend._310p.audio_encoder_acl_graph import (
    AUDIO_ENCODER_PROMPT_ATTENTION_ALIGNMENT,
    get_audio_encoder_prompt_attention_mask,
)

MIN_PAD_SIZE: int = 64  # min_size to pad weight
MAX_PAD_SIZE: int = 128  # max_size to pad weight

def is_approximate_calculation_supported() -> bool:
    return hasattr(torch_npu, "_npu_flash_attention_unpad_v2")


class AscendMMEncoderAttention310(MMEncoderAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MMEncoderAttention.
            multimodal_config: configs for multi-modal.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE
        self.scale_value = self.head_size**-0.5
        self.support_approximate_calculation = is_approximate_calculation_supported()

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor | None = None,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4
        prompt_mask = get_audio_encoder_prompt_attention_mask()

        # q, k, v: [b, s, head, head_dim] -> [b * s, head, head_dim]
        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        if self.enable_pad:
            origin_shape = q.shape[-1]
            pad_len = MAX_PAD_SIZE - origin_shape
            # [b * s, head, head_dim] -> [b * s, head, MAX_PAD_SIZE]
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        if prompt_mask is not None:
            if bsz != 1 or q_len != kv_len:
                raise ValueError("Audio graph requires batch-1 self-attention")
            attention_tokens = prompt_mask.shape[0]
            if (
                prompt_mask.ndim != 2
                or prompt_mask.shape[1] != attention_tokens
                or attention_tokens < q_len
                or attention_tokens % AUDIO_ENCODER_PROMPT_ATTENTION_ALIGNMENT
            ):
                raise ValueError("Invalid audio graph attention mask shape")
            if attention_tokens > q_len:
                padding = (0, 0, 0, 0, 0, attention_tokens - q_len)
                q, k, v = F.pad(q, padding), F.pad(k, padding), F.pad(v, padding)
            head_dim = q.shape[-1]

            def to_bnsd(tensor: torch.Tensor) -> torch.Tensor:
                return (
                    tensor.reshape(bsz, attention_tokens, self.num_heads, head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )

            context_layer = torch_npu.npu_prompt_flash_attention(
                to_bnsd(q),
                to_bnsd(k),
                to_bnsd(v),
                atten_mask=prompt_mask,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                scale_value=self.scale_value,
                pre_tokens=2147483647,
                next_tokens=2147483647,
                input_layout="BNSD",
                sparse_mode=0,
            )
            context_layer = (
                context_layer.transpose(1, 2)
                .contiguous()
                .view(bsz * attention_tokens, self.num_heads, head_dim)[: bsz * q_len]
            )
        else:
            if sequence_lengths is not None:
                if (
                    sequence_lengths.device.type == "cpu"
                    and sequence_lengths.dtype == torch.int32
                    and sequence_lengths.is_contiguous()
                ):
                    seq_lens_cpu = sequence_lengths
                else:
                    seq_lens_cpu = sequence_lengths.to(
                        device="cpu", dtype=torch.int32
                    ).contiguous()
            else:
                if cu_seqlens is None:
                    cu_seqlens = torch.arange(
                        0,
                        (bsz + 1) * q_len,
                        step=q_len,
                        dtype=torch.int32,
                        device="cpu",
                    )
                seq_lens_cpu = torch.diff(cu_seqlens).to("cpu")

            context_layer = torch.empty_like(q)
            if self.support_approximate_calculation:
                torch_npu._npu_flash_attention_unpad_v2(
                    query=q,
                    key=k,
                    value=v,
                    seq_len=seq_lens_cpu,
                    scale_value=self.scale_value,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    out=context_layer,
                    kernel_type=2,
                )
            else:
                torch_npu._npu_flash_attention_unpad(
                    query=q,
                    key=k,
                    value=v,
                    seq_len=seq_lens_cpu,
                    scale_value=self.scale_value,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    out=context_layer,
                )

        if self.enable_pad:
            context_layer = context_layer[..., :origin_shape]

        if is_reshaped:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz).contiguous()
        else:
            context_layer = einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz).contiguous()
        return context_layer

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

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
from tqdm import tqdm
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.platforms import current_platform


AUDIO_ENCODER_PROMPT_GRAPH_SIZE = 128
_PROMPT_ATTENTION_MASK: ContextVar[torch.Tensor | None] = ContextVar(
    "audio_encoder_prompt_attention_mask",
    default=None,
)


def get_audio_encoder_prompt_attention_mask() -> torch.Tensor | None:
    """Return the fixed Device mask while capturing an audio encoder graph."""

    return _PROMPT_ATTENTION_MASK.get()


@contextmanager
def use_audio_encoder_prompt_attention(
    mask: torch.Tensor,
) -> Iterator[None]:
    """Select PromptFlashAttention for one encoder-body capture."""

    token = _PROMPT_ATTENTION_MASK.set(mask)
    try:
        yield
    finally:
        _PROMPT_ATTENTION_MASK.reset(token)


def build_padded_attention_mask(
    sequence_lengths: Sequence[int],
    graph_size: int = AUDIO_ENCODER_PROMPT_GRAPH_SIZE,
) -> tuple[torch.Tensor, int]:
    """Build a block mask and isolate padding as an independent sequence.

    ``False`` entries are visible to PromptFlashAttention and ``True`` entries
    are masked. The returned mask stays on CPU so callers can reuse a fixed
    host staging buffer before copying it to the graph's fixed Device address.
    """

    topology = tuple(int(length) for length in sequence_lengths)
    if not topology or any(length <= 0 for length in topology):
        raise ValueError("audio encoder sequence lengths must be positive")
    actual_tokens = sum(topology)
    if actual_tokens > graph_size:
        raise ValueError(
            "audio encoder sequence lengths exceed graph size: "
            f"actual={actual_tokens}, graph_size={graph_size}"
        )

    dummy_tokens = graph_size - actual_tokens
    graph_topology = topology + ((dummy_tokens,) if dummy_tokens else ())
    mask = torch.ones((graph_size, graph_size), dtype=torch.bool, device="cpu")
    offset = 0
    for length in graph_topology:
        end = offset + length
        mask[offset:end, offset:end] = False
        offset = end
    return mask.contiguous(), dummy_tokens


class FixedAudioEncoderAclGraphRunner:
    """Fixed 128-token Qwen3-ASR encoder graph with a dynamic Device mask."""

    def __init__(
        self,
        encoder: Any,
        eager_forward: Callable[..., torch.Tensor],
        num_tokens: int,
        *,
        announce_enabled: bool = True,
    ) -> None:
        if num_tokens != AUDIO_ENCODER_PROMPT_GRAPH_SIZE:
            raise ValueError(
                "the 310P padded audio encoder prototype only supports "
                f"{AUDIO_ENCODER_PROMPT_GRAPH_SIZE} tokens"
            )
        self.encoder = encoder
        self.eager_forward = eager_forward
        self.num_tokens = num_tokens
        self.expected_sequence_lengths = (num_tokens,)
        self.graph: Any | None = None
        self.static_input: torch.Tensor | None = None
        self.static_cu_seqlens: torch.Tensor | None = None
        self.static_max_seqlen: torch.Tensor | None = None
        self.static_sequence_lengths: torch.Tensor | None = None
        self.static_attention_mask: torch.Tensor | None = None
        self.host_attention_mask: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None
        self.capture_failed = False
        self.capture_error: str | None = None
        self.last_action = "uninitialized"
        self._request_count = 0
        if announce_enabled:
            print(
                "[310P_AUDIO_GRAPH] PromptFlashAttention padding enabled: "
                f"size={num_tokens}",
                flush=True,
            )

    def _run_eager(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> torch.Tensor:
        return self.eager_forward(
            self.encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
            num_audios,
        )

    def _run_graph_body(self) -> torch.Tensor:
        assert self.static_input is not None
        assert self.static_cu_seqlens is not None
        assert self.static_sequence_lengths is not None
        assert self.static_attention_mask is not None
        with use_audio_encoder_prompt_attention(self.static_attention_mask):
            return self._run_eager(
                self.static_input,
                self.static_cu_seqlens,
                self.static_max_seqlen,
                self.static_sequence_lengths,
                1,
            )

    def _capture(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
    ) -> None:
        self.static_input = torch.empty_like(hidden_states)
        self.static_input.copy_(hidden_states)
        self.static_cu_seqlens = cu_seqlens.clone()
        self.static_max_seqlen = (
            None if max_seqlen is None else max_seqlen.clone()
        )
        self.static_sequence_lengths = sequence_lengths.clone()
        capture_mask, _ = build_padded_attention_mask((self.num_tokens,))
        self.host_attention_mask = torch.empty_like(capture_mask)
        self.host_attention_mask.copy_(capture_mask)
        self.static_attention_mask = capture_mask.to(hidden_states.device)

        for _ in range(2):
            self._run_graph_body()
        torch.npu.synchronize()

        self.graph = torch.npu.NPUGraph()
        graph_pool = current_platform.get_global_graph_pool()
        with torch.npu.graph(self.graph, pool=graph_pool):
            self.static_output = self._run_graph_body()

    def capture(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
    ) -> bool:
        if self.graph is not None:
            return True
        try:
            self._capture(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
            )
            torch.npu.synchronize()
            self.last_action = "startup_capture"
            return True
        except Exception as exc:
            self.capture_failed = True
            self.capture_error = f"{type(exc).__name__}: {exc}"
            self.graph = None
            self.static_input = None
            self.static_cu_seqlens = None
            self.static_max_seqlen = None
            self.static_sequence_lengths = None
            self.static_attention_mask = None
            self.host_attention_mask = None
            self.static_output = None
            self.last_action = "capture_failed"
            return False

    def _eligibility_error(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> tuple[str, str] | None:
        if self.capture_failed:
            return "capture_failed", ""
        if self.encoder.enforce_eager:
            return "enforce_eager", ""
        if get_tensor_model_parallel_world_size() != 1:
            return "unsupported_tp", ""
        if hidden_states.dtype != torch.float16:
            return "unsupported_dtype", f"actual={hidden_states.dtype}"
        if hidden_states.device.type != "npu":
            return "unsupported_device", f"actual={hidden_states.device}"
        if not hidden_states.is_contiguous():
            return "non_contiguous_input", ""
        if (
            sequence_lengths.device.type != "cpu"
            or sequence_lengths.dtype != torch.int32
            or not sequence_lengths.is_contiguous()
        ):
            return "invalid_sequence_lengths", ""

        topology = tuple(sequence_lengths.tolist())
        actual_tokens = hidden_states.shape[0]
        if not topology or any(length <= 0 for length in topology):
            return "invalid_topology", f"seq_lens={list(topology)}"
        if sum(topology) != actual_tokens:
            return (
                "invalid_topology",
                f"tokens={actual_tokens}, seq_lens={list(topology)}",
            )
        if actual_tokens > self.num_tokens:
            return (
                "token_overflow",
                f"actual={actual_tokens}, limit={self.num_tokens}",
            )
        if self.graph is None:
            return "graph_not_captured", ""
        return None

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
        num_audios: int,
        *,
        request_id: int | None = None,
    ) -> torch.Tensor:
        if request_id is None:
            self._request_count += 1
            request_id = self._request_count

        error = self._eligibility_error(hidden_states, sequence_lengths)
        if error is not None:
            reason, detail = error
            suffix = f", {detail}" if detail else ""
            print(
                f"[310P_AUDIO_GRAPH] request={request_id}, graph_hit=False, "
                f"reason={reason}{suffix}",
                flush=True,
            )
            self.last_action = "fallback"
            return self._run_eager(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        assert self.static_input is not None
        assert self.static_attention_mask is not None
        assert self.host_attention_mask is not None
        assert self.static_output is not None

        topology = tuple(sequence_lengths.tolist())
        actual_tokens = hidden_states.shape[0]
        mask, dummy_tokens = build_padded_attention_mask(
            topology,
            self.num_tokens,
        )
        self.static_input[:actual_tokens].copy_(hidden_states)
        if dummy_tokens:
            self.static_input[actual_tokens:].zero_()
        self.host_attention_mask.copy_(mask)
        self.static_attention_mask.copy_(
            self.host_attention_mask,
            non_blocking=True,
        )
        self.graph.replay()
        self.last_action = "replay"
        print(
            f"[310P_AUDIO_GRAPH] request={request_id}, graph_hit=True, "
            f"actual={actual_tokens}, padded={self.num_tokens}, "
            f"seq_lens={list(topology)}, dummy={dummy_tokens}",
            flush=True,
        )
        return self.static_output[:actual_tokens].clone()


class AudioEncoderAclGraphPool:
    """The first-stage 310P padded audio encoder graph experiment."""

    def __init__(
        self,
        encoder: Any,
        eager_forward: Callable[..., torch.Tensor],
        graph_sizes: Sequence[int],
    ) -> None:
        sizes = tuple(sorted(set(graph_sizes), reverse=True))
        if sizes != (AUDIO_ENCODER_PROMPT_GRAPH_SIZE,):
            raise ValueError(
                "the 310P padded audio encoder prototype requires "
                "audio_encoder_aclgraph_sizes=[128]"
            )
        self.encoder = encoder
        self.eager_forward = eager_forward
        self.graph_sizes = sizes
        self.runners = {
            AUDIO_ENCODER_PROMPT_GRAPH_SIZE: FixedAudioEncoderAclGraphRunner(
                encoder,
                eager_forward,
                AUDIO_ENCODER_PROMPT_GRAPH_SIZE,
                announce_enabled=False,
            )
        }
        self._request_count = 0
        print(
            "[310P_AUDIO_GRAPH] PromptFlashAttention padding enabled: "
            f"size={AUDIO_ENCODER_PROMPT_GRAPH_SIZE}",
            flush=True,
        )

    @property
    def runner(self) -> FixedAudioEncoderAclGraphRunner:
        return self.runners[AUDIO_ENCODER_PROMPT_GRAPH_SIZE]

    def _capture_runner(self) -> bool:
        hidden_size = int(self.encoder.ln_post.normalized_shape[0])
        hidden_states = torch.zeros(
            (AUDIO_ENCODER_PROMPT_GRAPH_SIZE, hidden_size),
            dtype=self.encoder.dtype,
            device=self.encoder.device,
        )
        sequence_lengths = torch.tensor(
            [AUDIO_ENCODER_PROMPT_GRAPH_SIZE],
            dtype=torch.int32,
            device="cpu",
        ).contiguous()
        cu_seqlens = torch.tensor(
            [0, AUDIO_ENCODER_PROMPT_GRAPH_SIZE],
            dtype=torch.int32,
            device=self.encoder.device,
        )
        max_seqlen = self.encoder.compute_attn_mask_seqlen(cu_seqlens)
        return self.runner.capture(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
        )

    def capture_all(self, *, show_progress: bool) -> tuple[int, ...]:
        if (
            self.encoder.enforce_eager
            or get_tensor_model_parallel_world_size() != 1
            or self.encoder.dtype != torch.float16
            or self.encoder.device.type != "npu"
        ):
            return ()
        with tqdm(
            (AUDIO_ENCODER_PROMPT_GRAPH_SIZE,),
            desc="Capturing padded audio encoder graph",
            unit="graph",
            disable=not show_progress,
        ) as progress:
            for size in progress:
                if show_progress:
                    progress.set_postfix(tokens=size)
                if not self._capture_runner():
                    return ()
        return (AUDIO_ENCODER_PROMPT_GRAPH_SIZE,)

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> torch.Tensor:
        self._request_count += 1
        return self.runner.run(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
            num_audios,
            request_id=self._request_count,
        )

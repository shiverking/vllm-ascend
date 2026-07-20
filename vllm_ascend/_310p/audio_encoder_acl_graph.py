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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch
from tqdm import tqdm
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.platforms import current_platform


def _get_attention_window_tokens(encoder: Any) -> int:
    chunk_size = encoder.n_window * 2
    chunk_tokens = chunk_size
    for _ in range(3):
        chunk_tokens = (chunk_tokens - 1) // 2 + 1
    window_tokens = chunk_tokens * (encoder.n_window_infer // chunk_size)
    if window_tokens <= 0:
        raise ValueError("audio encoder attention window must be positive")
    return window_tokens


@dataclass(frozen=True)
class AudioGraphChunk:
    runner: "FixedAudioEncoderAclGraphRunner"
    sequence_end: int
    token_start: int
    token_end: int


class FixedAudioEncoderAclGraphRunner:
    """One fixed-token, fixed-topology Qwen3-ASR encoder body graph."""

    def __init__(
        self,
        encoder: Any,
        eager_forward: Callable[..., torch.Tensor],
        num_tokens: int,
    ) -> None:
        if num_tokens <= 0:
            raise ValueError("audio encoder ACLGraph token size must be positive")
        self.encoder = encoder
        self.eager_forward = eager_forward
        self.num_tokens = num_tokens
        self.expected_sequence_lengths = self._build_expected_sequence_lengths()
        self.graph: Any | None = None
        self.static_input: torch.Tensor | None = None
        self.static_cu_seqlens: torch.Tensor | None = None
        self.static_max_seqlen: torch.Tensor | None = None
        self.static_sequence_lengths: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None

    def _build_expected_sequence_lengths(self) -> tuple[int, ...]:
        window_tokens = _get_attention_window_tokens(self.encoder)
        full_windows, remainder = divmod(self.num_tokens, window_tokens)
        result = [window_tokens] * full_windows
        if remainder:
            result.append(remainder)
        return tuple(result)

    def _run_eager(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.eager_forward(
            self.encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
            1,
        )

    def capture(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
    ) -> None:
        if self.graph is not None:
            return

        self.static_input = torch.empty_like(hidden_states)
        self.static_input.copy_(hidden_states)
        self.static_cu_seqlens = cu_seqlens.clone()
        self.static_max_seqlen = (
            None if max_seqlen is None else max_seqlen.clone()
        )
        self.static_sequence_lengths = sequence_lengths.clone()

        for _ in range(2):
            self._run_eager(
                self.static_input,
                self.static_cu_seqlens,
                self.static_max_seqlen,
                self.static_sequence_lengths,
            )
        torch.npu.synchronize()

        self.graph = torch.npu.NPUGraph()
        graph_pool = current_platform.get_global_graph_pool()
        with torch.npu.graph(self.graph, pool=graph_pool):
            self.static_output = self._run_eager(
                self.static_input,
                self.static_cu_seqlens,
                self.static_max_seqlen,
                self.static_sequence_lengths,
            )
        torch.npu.synchronize()

    def replay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError(
                f"audio encoder ACLGraph for {self.num_tokens} tokens "
                "was not captured"
            )
        assert self.static_input is not None
        assert self.static_output is not None
        self.static_input.copy_(hidden_states)
        self.graph.replay()
        return self.static_output.clone()


class AudioEncoderAclGraphPool:
    """Run sequence-aligned encoder chunks with fixed startup graphs."""

    def __init__(
        self,
        encoder: Any,
        eager_forward: Callable[..., torch.Tensor],
        graph_sizes: Sequence[int],
    ) -> None:
        sizes = tuple(sorted(set(graph_sizes), reverse=True))
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("audio encoder ACLGraph sizes must be positive")

        self.window_tokens = _get_attention_window_tokens(encoder)
        unaligned_sizes = [
            size for size in sizes if size % self.window_tokens != 0
        ]
        if unaligned_sizes:
            raise ValueError(
                "audio_encoder_aclgraph_sizes must contain complete attention "
                f"windows and be multiples of {self.window_tokens}; "
                f"unaligned sizes: {unaligned_sizes}"
            )

        self.encoder = encoder
        self.eager_forward = eager_forward
        self.graph_sizes = sizes
        self.runners = {
            size: FixedAudioEncoderAclGraphRunner(
                encoder,
                eager_forward,
                size,
            )
            for size in sizes
        }
        self._request_count = 0
        topologies = {
            size: list(self.runners[size].expected_sequence_lengths)
            for size in sizes
        }
        logger.info(
            "[310P_AUDIO_GRAPH] pool enabled: sizes=%s, topologies=%s",
            list(sizes),
            topologies,
        )

    @staticmethod
    def _make_cu_seqlens(topology: Sequence[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *accumulate(topology)],
            dtype=torch.int32,
            device="cpu",
        ).contiguous()

    def _capture_runner(self, runner: FixedAudioEncoderAclGraphRunner) -> None:
        topology = runner.expected_sequence_lengths
        hidden_size = int(self.encoder.ln_post.normalized_shape[0])
        hidden_states = torch.zeros(
            (runner.num_tokens, hidden_size),
            dtype=self.encoder.dtype,
            device=self.encoder.device,
        )
        sequence_lengths = torch.tensor(
            topology,
            dtype=torch.int32,
            device="cpu",
        ).contiguous()
        cu_seqlens = self._make_cu_seqlens(topology).to(self.encoder.device)
        max_seqlen = self.encoder.compute_attn_mask_seqlen(cu_seqlens)
        runner.capture(
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

        captured = []
        capture_sizes = tuple(reversed(self.graph_sizes))
        with tqdm(
            capture_sizes,
            desc="Capturing audio encoder graphs",
            unit="graph",
            disable=not show_progress,
        ) as progress:
            for size in progress:
                if show_progress:
                    progress.set_postfix(tokens=size)
                try:
                    self._capture_runner(self.runners[size])
                except Exception as exc:
                    raise RuntimeError(
                        "Audio encoder ACLGraph capture failed for "
                        f"size {size}: {type(exc).__name__}: {exc}"
                    ) from exc
                captured.append(size)

        logger.info(
            "[310P_AUDIO_GRAPH] capture complete: sizes=%s",
            captured,
        )
        return tuple(captured)

    def _build_plan(
        self,
        topology: Sequence[int],
    ) -> tuple[list[AudioGraphChunk], int, int]:
        topology = tuple(topology)
        chunks: list[AudioGraphChunk] = []
        sequence_index = 0
        token_offset = 0

        while sequence_index < len(topology):
            matched_runner = None
            for size in self.graph_sizes:
                runner = self.runners[size]
                if runner.graph is None:
                    continue
                expected = runner.expected_sequence_lengths
                sequence_end = sequence_index + len(expected)
                if tuple(topology[sequence_index:sequence_end]) == expected:
                    matched_runner = runner
                    break
            if matched_runner is None:
                break

            sequence_end = sequence_index + len(
                matched_runner.expected_sequence_lengths
            )
            token_end = token_offset + matched_runner.num_tokens
            chunks.append(
                AudioGraphChunk(
                    runner=matched_runner,
                    sequence_end=sequence_end,
                    token_start=token_offset,
                    token_end=token_end,
                )
            )
            sequence_index = sequence_end
            token_offset = token_end

        return chunks, sequence_index, token_offset

    def _check_eligibility(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> str | None:
        if self.encoder.enforce_eager:
            return "enforce_eager"
        if num_audios != 1:
            return f"multi_audio, actual={num_audios}"
        if get_tensor_model_parallel_world_size() != 1:
            return "unsupported_tp"
        if hidden_states.dtype != torch.float16:
            return f"unsupported_dtype, actual={hidden_states.dtype}"
        if hidden_states.device.type != "npu":
            return f"unsupported_device, actual={hidden_states.device}"
        if not hidden_states.is_contiguous():
            return "non_contiguous_input"
        if (
            sequence_lengths.device.type != "cpu"
            or sequence_lengths.dtype != torch.int32
            or not sequence_lengths.is_contiguous()
        ):
            return "invalid_sequence_lengths"
        if sum(sequence_lengths.tolist()) != hidden_states.shape[0]:
            return "invalid_topology"
        return None

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

    def _run_eager_suffix(
        self,
        hidden_states: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        topology: tuple[int, ...],
        token_start: int,
        sequence_start: int,
    ) -> torch.Tensor:
        suffix_topology = topology[sequence_start:]
        suffix_sequence_lengths = torch.tensor(
            suffix_topology,
            dtype=torch.int32,
            device="cpu",
        ).contiguous()
        suffix_cu_seqlens = self._make_cu_seqlens(suffix_topology).to(
            hidden_states.device
        )
        suffix_max_seqlen = None
        if max_seqlen is not None:
            suffix_max_seqlen = torch.tensor(
                max(suffix_topology),
                dtype=max_seqlen.dtype,
                device=max_seqlen.device,
            )
        return self._run_eager(
            hidden_states[token_start:],
            suffix_cu_seqlens,
            suffix_max_seqlen,
            suffix_sequence_lengths,
            1,
        )

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> torch.Tensor:
        self._request_count += 1
        request_id = self._request_count
        reason = self._check_eligibility(
            hidden_states,
            sequence_lengths,
            num_audios,
        )
        if reason is not None:
            topology = (
                sequence_lengths.tolist()
                if sequence_lengths.device.type == "cpu"
                else "non_cpu"
            )
            logger.info(
                "[310P_AUDIO_GRAPH] request=%d, graph_hit=False, "
                "tokens=%d, seq_lens=%s, fallback=%s",
                request_id,
                hidden_states.shape[0],
                topology,
                reason,
            )
            return self._run_eager(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        topology = tuple(sequence_lengths.tolist())
        chunks, tail_sequence_start, tail_token_start = self._build_plan(topology)
        if not chunks:
            logger.info(
                "[310P_AUDIO_GRAPH] request=%d, graph_hit=False, "
                "tokens=%d, seq_lens=%s, fallback=no_matching_graph",
                request_id,
                hidden_states.shape[0],
                list(topology),
            )
            return self._run_eager(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        outputs = [
            chunk.runner.replay(hidden_states[chunk.token_start : chunk.token_end])
            for chunk in chunks
        ]
        eager_tail_tokens = hidden_states.shape[0] - tail_token_start
        if eager_tail_tokens:
            outputs.append(
                self._run_eager_suffix(
                    hidden_states,
                    max_seqlen,
                    topology,
                    tail_token_start,
                    tail_sequence_start,
                )
            )

        chunk_sizes = [chunk.runner.num_tokens for chunk in chunks]
        logger.info(
            "[310P_AUDIO_GRAPH] request=%d, graph_hit=True, tokens=%d, "
            "seq_lens=%s, graph_chunks=%s, eager_tail_tokens=%d",
            request_id,
            hidden_states.shape[0],
            list(topology),
            chunk_sizes,
            eager_tail_tokens,
        )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

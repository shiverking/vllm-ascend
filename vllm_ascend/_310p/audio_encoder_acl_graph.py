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
    sequence_start: int
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
        *,
        announce_enabled: bool = True,
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
        self.capture_failed = False
        self.capture_error: str | None = None
        self.last_action = "uninitialized"
        self._log_keys: set[str] = set()
        self._request_count = 0
        self._active_request_id = 0
        self._active_chunk_index: int | None = None
        self._active_chunk_count: int | None = None
        if announce_enabled:
            self._print_once("enabled", f"enabled: tokens={num_tokens}")

    def _build_expected_sequence_lengths(self) -> tuple[int, ...]:
        window_tokens = _get_attention_window_tokens(self.encoder)
        full_windows, remainder = divmod(self.num_tokens, window_tokens)
        result = [window_tokens] * full_windows
        if remainder:
            result.append(remainder)
        return tuple(result)

    def _print_once(self, key: str, message: str) -> None:
        if key in self._log_keys:
            return
        self._log_keys.add(key)
        print(f"[310P_AUDIO_GRAPH] {message}", flush=True)

    def _print_request_status(
        self,
        graph_hit: bool,
        action: str,
        detail: str = "",
    ) -> None:
        chunk = ""
        if self._active_chunk_index is not None:
            chunk = (
                f", chunk={self._active_chunk_index}/{self._active_chunk_count}"
                f", graph_size={self.num_tokens}"
            )
        suffix = f", {detail}" if detail else ""
        print(
            f"[310P_AUDIO_GRAPH] request={self._active_request_id}, "
            f"graph_hit={graph_hit}, action={action}{chunk}{suffix}",
            flush=True,
        )

    def _fallback(self, reason: str, detail: str = "") -> None:
        self.last_action = "fallback"
        suffix = f", {detail}" if detail else ""
        self._print_request_status(
            graph_hit=False,
            action="fallback",
            detail=f"reason={reason}{suffix}",
        )

    def _check_eligibility(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> bool:
        if self.capture_failed:
            self._fallback("capture_failed")
            return False
        if self.encoder.enforce_eager:
            self._fallback("enforce_eager")
            return False
        if num_audios != 1:
            self._fallback("multi_audio", f"actual={num_audios}")
            return False
        if get_tensor_model_parallel_world_size() != 1:
            self._fallback("unsupported_tp")
            return False
        if hidden_states.dtype != torch.float16:
            self._fallback("unsupported_dtype", f"actual={hidden_states.dtype}")
            return False
        if hidden_states.device.type != "npu":
            self._fallback("unsupported_device", f"actual={hidden_states.device}")
            return False
        if not hidden_states.is_contiguous():
            self._fallback("non_contiguous_input")
            return False
        if (
            sequence_lengths.device.type != "cpu"
            or sequence_lengths.dtype != torch.int32
            or not sequence_lengths.is_contiguous()
        ):
            self._fallback("invalid_sequence_lengths")
            return False

        actual_tokens = hidden_states.shape[0]
        actual_topology = tuple(sequence_lengths.tolist())
        if actual_tokens != self.num_tokens:
            self._fallback(
                "token_mismatch",
                f"actual={actual_tokens}, expected={self.num_tokens}, "
                f"seq_lens={list(actual_topology)}",
            )
            return False
        if actual_topology != self.expected_sequence_lengths:
            self._fallback(
                "topology_mismatch",
                f"actual={list(actual_topology)}, "
                f"expected={list(self.expected_sequence_lengths)}",
            )
            return False
        return True

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
        self.static_max_seqlen = None if max_seqlen is None else max_seqlen.clone()
        self.static_sequence_lengths = sequence_lengths.clone()

        for _ in range(2):
            self._run_eager(
                self.static_input,
                self.static_cu_seqlens,
                self.static_max_seqlen,
                self.static_sequence_lengths,
                1,
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
                1,
            )

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
            self.static_output = None
            self.last_action = "capture_failed"
            return False

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
        num_audios: int,
        *,
        request_id: int | None = None,
        chunk_index: int | None = None,
        chunk_count: int | None = None,
    ) -> torch.Tensor:
        if request_id is None:
            self._request_count += 1
            request_id = self._request_count
        self._active_request_id = request_id
        self._active_chunk_index = chunk_index
        self._active_chunk_count = chunk_count

        if not self._check_eligibility(
            hidden_states, sequence_lengths, num_audios
        ):
            return self._run_eager(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        if self.graph is None:
            self._fallback("graph_not_captured")
            return self._run_eager(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        assert self.static_input is not None
        assert self.static_output is not None
        self.static_input.copy_(hidden_states)
        self.last_action = "replay"
        self._print_request_status(
            graph_hit=True,
            action="replay",
            detail=(
                f"tokens={hidden_states.shape[0]}, "
                f"seq_lens={sequence_lengths.tolist()}"
            ),
        )
        self.graph.replay()
        return self.static_output.clone()


class AudioEncoderAclGraphPool:
    """Greedily run encoder sequence-aligned chunks with fixed graphs."""

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
                announce_enabled=False,
            )
            for size in sizes
        }
        self._request_count = 0
        self._sequence_lengths_cache: dict[int, torch.Tensor] = {}
        self._cu_seqlens_cache: dict[int, torch.Tensor] = {}
        self._max_seqlen_cache: dict[int, torch.Tensor] = {}
        topologies = {
            size: list(self.runners[size].expected_sequence_lengths)
            for size in sizes
        }
        print(
            f"[310P_AUDIO_GRAPH] pool enabled: sizes={list(sizes)}, "
            f"topologies={topologies}",
            flush=True,
        )

    def _capture_runner(self, runner: FixedAudioEncoderAclGraphRunner) -> bool:
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
        return runner.capture(
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
                runner = self.runners[size]
                try:
                    success = self._capture_runner(runner)
                except Exception as exc:
                    runner.capture_failed = True
                    runner.capture_error = f"{type(exc).__name__}: {exc}"
                    success = False
                if not success:
                    raise RuntimeError(
                        "Audio encoder ACLGraph capture failed for "
                        f"size {size}: {runner.capture_error}"
                    )
                captured.append(size)
        return tuple(captured)

    @staticmethod
    def _make_cu_seqlens(topology: Sequence[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *accumulate(topology)],
            dtype=torch.int32,
            device="cpu",
        ).contiguous()

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
                if runner.capture_failed:
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
                    sequence_start=sequence_index,
                    sequence_end=sequence_end,
                    token_start=token_offset,
                    token_end=token_end,
                )
            )
            sequence_index = sequence_end
            token_offset = token_end

        return chunks, sequence_index, token_offset

    def _check_common_eligibility(
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

    def _get_graph_metadata(
        self,
        runner: FixedAudioEncoderAclGraphRunner,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        size = runner.num_tokens
        if size not in self._sequence_lengths_cache:
            topology = runner.expected_sequence_lengths
            sequence_lengths = torch.tensor(
                topology,
                dtype=torch.int32,
                device="cpu",
            ).contiguous()
            self._sequence_lengths_cache[size] = sequence_lengths
            self._cu_seqlens_cache[size] = self._make_cu_seqlens(topology).to(
                cu_seqlens.device
            )
            if max_seqlen is not None:
                self._max_seqlen_cache[size] = torch.tensor(
                    max(topology),
                    dtype=max_seqlen.dtype,
                    device=max_seqlen.device,
                )
        return (
            self._cu_seqlens_cache[size],
            self._max_seqlen_cache.get(size),
            self._sequence_lengths_cache[size],
        )

    def _run_eager_suffix(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
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
            cu_seqlens.device
        )
        suffix_max_seqlen = None
        if max_seqlen is not None:
            suffix_max_seqlen = torch.tensor(
                max(suffix_topology),
                dtype=max_seqlen.dtype,
                device=max_seqlen.device,
            )
        return self.eager_forward(
            self.encoder,
            hidden_states[token_start:],
            suffix_cu_seqlens,
            suffix_max_seqlen,
            suffix_sequence_lengths,
            1,
        )

    def _print_whole_fallback(
        self,
        reason: str,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> None:
        topology = (
            sequence_lengths.tolist()
            if sequence_lengths.device.type == "cpu"
            else "non_cpu"
        )
        print(
            f"[310P_AUDIO_GRAPH] request={self._request_count}, "
            f"graph_hit=False, action=fallback, reason={reason}, "
            f"tokens={hidden_states.shape[0]}, seq_lens={topology}",
            flush=True,
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
        reason = self._check_common_eligibility(
            hidden_states, sequence_lengths, num_audios
        )
        if reason is not None:
            self._print_whole_fallback(reason, hidden_states, sequence_lengths)
            return self.eager_forward(
                self.encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        topology = tuple(sequence_lengths.tolist())
        chunks, tail_sequence_start, tail_token_start = self._build_plan(topology)
        chunk_sizes = [chunk.runner.num_tokens for chunk in chunks]
        eager_tail_tokens = hidden_states.shape[0] - tail_token_start
        print(
            f"[310P_AUDIO_GRAPH] request={request_id}, action=plan, "
            f"tokens={hidden_states.shape[0]}, seq_lens={list(topology)}, "
            f"graph_chunks={chunk_sizes}, "
            f"eager_tail_tokens={eager_tail_tokens}",
            flush=True,
        )

        if not chunks:
            print(
                f"[310P_AUDIO_GRAPH] request={request_id}, graph_hit=False, "
                f"action=eager_tail, tokens={hidden_states.shape[0]}, "
                f"seq_lens={list(topology)}",
                flush=True,
            )
            output = self.eager_forward(
                self.encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )
            print(
                f"[310P_AUDIO_GRAPH] request={request_id}, graph_hit=False, "
                f"action=complete, graph_chunks=0, captures=0, "
                f"replay_hits=0, fallback_chunks=0, "
                f"eager_tail_tokens={hidden_states.shape[0]}",
                flush=True,
            )
            return output

        outputs: list[torch.Tensor] = []
        captures = 0
        replay_hits = 0
        fallback_chunks = 0
        executed_chunks = 0
        for index, chunk in enumerate(chunks, start=1):
            chunk_cu, chunk_max, chunk_sequence_lengths = self._get_graph_metadata(
                chunk.runner,
                cu_seqlens,
                max_seqlen,
            )
            output = chunk.runner.run(
                hidden_states[chunk.token_start : chunk.token_end],
                chunk_cu,
                chunk_max,
                chunk_sequence_lengths,
                1,
                request_id=request_id,
                chunk_index=index,
                chunk_count=len(chunks),
            )
            outputs.append(output)
            executed_chunks += 1
            if chunk.runner.last_action == "capture":
                captures += 1
            elif chunk.runner.last_action == "replay":
                replay_hits += 1
            else:
                fallback_chunks += 1
                tail_sequence_start = chunk.sequence_end
                tail_token_start = chunk.token_end
                break

        remaining_tail_tokens = hidden_states.shape[0] - tail_token_start
        if remaining_tail_tokens:
            tail_topology = topology[tail_sequence_start:]
            print(
                f"[310P_AUDIO_GRAPH] request={request_id}, graph_hit=False, "
                f"action=eager_tail, tokens={remaining_tail_tokens}, "
                f"seq_lens={list(tail_topology)}",
                flush=True,
            )
            outputs.append(
                self._run_eager_suffix(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    topology,
                    tail_token_start,
                    tail_sequence_start,
                )
            )

        print(
            f"[310P_AUDIO_GRAPH] request={request_id}, "
            f"graph_hit={replay_hits > 0}, action=complete, "
            f"graph_chunks={executed_chunks}, captures={captures}, "
            f"replay_hits={replay_hits}, fallback_chunks={fallback_chunks}, "
            f"eager_tail_tokens={remaining_tail_tokens}",
            flush=True,
        )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

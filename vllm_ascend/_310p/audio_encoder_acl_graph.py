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

from collections.abc import Callable
from typing import Any

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.platforms import current_platform


class FixedAudioEncoderAclGraphRunner:
    """Experimental fixed-shape ACLGraph runner for a Qwen3-ASR encoder body."""

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
        self.capture_failed = False
        self._log_keys: set[str] = set()
        self._print_once("enabled", f"enabled: tokens={num_tokens}")

    def _build_expected_sequence_lengths(self) -> tuple[int, ...]:
        chunk_size = self.encoder.n_window * 2
        chunk_tokens = chunk_size
        for _ in range(3):
            chunk_tokens = (chunk_tokens - 1) // 2 + 1
        window_tokens = chunk_tokens * (self.encoder.n_window_infer // chunk_size)
        if window_tokens <= 0:
            raise ValueError("audio encoder attention window must be positive")
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

    def _fallback(self, reason: str, detail: str = "") -> None:
        suffix = f", {detail}" if detail else ""
        self._print_once(f"fallback:{reason}", f"fallback: reason={reason}{suffix}")

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
        actual_tokens = hidden_states.shape[0]
        if actual_tokens != self.num_tokens:
            self._fallback(
                "token_mismatch",
                f"actual={actual_tokens}, expected={self.num_tokens}",
            )
            return False
        if (
            sequence_lengths.device.type != "cpu"
            or sequence_lengths.dtype != torch.int32
            or not sequence_lengths.is_contiguous()
        ):
            self._fallback("invalid_sequence_lengths")
            return False
        actual_topology = tuple(sequence_lengths.tolist())
        if actual_topology != self.expected_sequence_lengths:
            self._fallback(
                "topology_mismatch",
                f"actual={list(actual_topology)}, expected={list(self.expected_sequence_lengths)}",
            )
            return False
        self._print_once(
            "eligible",
            f"runtime eligible: tokens={actual_tokens}, seq_lens={list(actual_topology)}",
        )
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
    ) -> torch.Tensor:
        self.static_input = torch.empty_like(hidden_states)
        self.static_input.copy_(hidden_states)
        self.static_cu_seqlens = cu_seqlens.clone()
        self.static_max_seqlen = None if max_seqlen is None else max_seqlen.clone()
        self.static_sequence_lengths = sequence_lengths.clone()

        self._print_once("warmup", "warmup begin")
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
        self._print_once("capture_begin", "capture begin")
        with torch.npu.graph(self.graph, pool=graph_pool):
            self.static_output = self._run_eager(
                self.static_input,
                self.static_cu_seqlens,
                self.static_max_seqlen,
                self.static_sequence_lengths,
                1,
            )
        self._print_once("capture_complete", "capture complete")
        return self.static_output.clone()

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> torch.Tensor:
        if not self._check_eligibility(hidden_states, sequence_lengths, num_audios):
            return self._run_eager(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        if self.graph is None:
            try:
                return self._capture(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    sequence_lengths,
                )
            except Exception as exc:
                self.capture_failed = True
                self.graph = None
                self.static_output = None
                self._fallback("capture_error", f"error={type(exc).__name__}: {exc}")
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
        self._print_once("replay_begin", "replay begin")
        self.graph.replay()
        self._print_once("replay_complete", "replay complete")
        return self.static_output.clone()

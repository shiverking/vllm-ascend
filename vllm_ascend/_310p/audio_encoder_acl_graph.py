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
from itertools import accumulate
from typing import Any

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.platforms import current_platform


MAX_PADDING_TOKENS = 4
MAX_REAL_SEQUENCE_COUNT = 2
STATIC_SEQUENCE_COUNT = 3
SHAPE_OBSERVATION_LIMIT = 4


class FixedAudioEncoderAclGraphRunner:
    """Experimental padded ACLGraph runner for a Qwen3-ASR encoder body."""

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
        self.graph: Any | None = None
        self.static_input: torch.Tensor | None = None
        self.static_cu_seqlens: torch.Tensor | None = None
        self.static_cu_seqlens_cpu: torch.Tensor | None = None
        self.static_max_seqlen: torch.Tensor | None = None
        self.static_max_seqlen_cpu: torch.Tensor | None = None
        self.static_sequence_lengths: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None
        self.capture_failed = False
        self._log_keys: set[str] = set()
        self._shape_observation_count = 0
        self._host_topology_address: int | None = None
        self._print_once("enabled", f"enabled: tokens={num_tokens}")

    @staticmethod
    def _split_padding(padding_tokens: int, sequence_count: int) -> tuple[int, ...]:
        if sequence_count <= 0 or padding_tokens < sequence_count:
            return ()
        base, remainder = divmod(padding_tokens, sequence_count)
        return tuple(
            base + (1 if index >= sequence_count - remainder else 0)
            for index in range(sequence_count)
        )

    def _build_graph_topology(
        self,
        real_topology: Sequence[int],
        actual_tokens: int,
    ) -> tuple[int, ...] | None:
        real_topology = tuple(int(length) for length in real_topology)
        if (
            not real_topology
            or len(real_topology) > MAX_REAL_SEQUENCE_COUNT
            or any(length <= 0 for length in real_topology)
            or sum(real_topology) != actual_tokens
        ):
            return None

        padding_tokens = self.num_tokens - actual_tokens
        dummy_count = STATIC_SEQUENCE_COUNT - len(real_topology)
        dummy_topology = self._split_padding(padding_tokens, dummy_count)
        if len(dummy_topology) != dummy_count:
            return None
        return real_topology + dummy_topology

    def _print_once(self, key: str, message: str) -> None:
        if key in self._log_keys:
            return
        self._log_keys.add(key)
        print(f"[310P_AUDIO_GRAPH] {message}", flush=True)

    def _fallback(self, reason: str, detail: str = "") -> None:
        suffix = f", {detail}" if detail else ""
        self._print_once(f"fallback:{reason}", f"fallback: reason={reason}{suffix}")

    def _observe_shape(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> None:
        if self._shape_observation_count >= SHAPE_OBSERVATION_LIMIT:
            return
        topology = (
            sequence_lengths.tolist()
            if sequence_lengths.device.type == "cpu"
            else "non_cpu"
        )
        print(
            "[310P_AUDIO_GRAPH] shape observation: "
            f"actual_tokens={hidden_states.shape[0]}, num_audios={num_audios}, "
            f"sequence_lengths={topology}, "
            f"sequence_count={sequence_lengths.numel()}",
            flush=True,
        )
        self._shape_observation_count += 1

    def _check_eligibility(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> tuple[int, ...] | None:
        self._observe_shape(hidden_states, sequence_lengths, num_audios)
        if self.capture_failed:
            self._fallback("capture_failed")
            return None
        if self.encoder.enforce_eager:
            self._fallback("enforce_eager")
            return None
        if get_tensor_model_parallel_world_size() != 1:
            self._fallback("unsupported_tp")
            return None
        if hidden_states.dtype != torch.float16:
            self._fallback("unsupported_dtype", f"actual={hidden_states.dtype}")
            return None
        if hidden_states.device.type != "npu":
            self._fallback("unsupported_device", f"actual={hidden_states.device}")
            return None
        if not hidden_states.is_contiguous():
            self._fallback("non_contiguous_input")
            return None
        if (
            sequence_lengths.device.type != "cpu"
            or sequence_lengths.dtype != torch.int32
            or not sequence_lengths.is_contiguous()
        ):
            self._fallback("invalid_topology", "sequence_lengths must be CPU int32 contiguous")
            return None

        actual_tokens = hidden_states.shape[0]
        if actual_tokens == self.num_tokens:
            self._fallback("exact_graph_size", f"actual={actual_tokens}")
            return None
        if not self.num_tokens - MAX_PADDING_TOKENS <= actual_tokens < self.num_tokens:
            self._fallback(
                "padding_ratio",
                f"actual={actual_tokens}, padded={self.num_tokens}",
            )
            return None

        real_topology = tuple(sequence_lengths.tolist())
        if len(real_topology) > MAX_REAL_SEQUENCE_COUNT:
            self._fallback("too_many_sequences", f"actual={len(real_topology)}")
            return None
        graph_topology = self._build_graph_topology(real_topology, actual_tokens)
        if graph_topology is None:
            self._fallback("invalid_topology", f"actual={list(real_topology)}")
            return None

        self._print_once(
            "padding_eligible",
            f"padding eligible: actual={actual_tokens}, padded={self.num_tokens}",
        )
        self._print_once(
            "topology",
            f"topology: real={list(real_topology)}, graph={list(graph_topology)}",
        )
        return graph_topology

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

    @staticmethod
    def _make_cu_seqlens(topology: Sequence[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *accumulate(topology)],
            dtype=torch.int32,
            device="cpu",
        ).contiguous()

    def _prepare_static_inputs(
        self,
        hidden_states: torch.Tensor,
        graph_topology: tuple[int, ...],
    ) -> None:
        actual_tokens = hidden_states.shape[0]
        if self.static_input is None:
            self.static_input = hidden_states.new_zeros(
                (self.num_tokens, *hidden_states.shape[1:])
            )
            self.static_sequence_lengths = torch.empty(
                STATIC_SEQUENCE_COUNT,
                dtype=torch.int32,
                device="cpu",
            ).contiguous()
            self.static_cu_seqlens_cpu = torch.empty(
                STATIC_SEQUENCE_COUNT + 1,
                dtype=torch.int32,
                device="cpu",
            ).contiguous()
            self.static_cu_seqlens = self.static_cu_seqlens_cpu.to(
                hidden_states.device
            )
            self._host_topology_address = self.static_sequence_lengths.data_ptr()

        assert self.static_sequence_lengths is not None
        assert self.static_cu_seqlens_cpu is not None
        assert self.static_cu_seqlens is not None
        self.static_input.zero_()
        self.static_input[:actual_tokens].copy_(hidden_states)
        self.static_sequence_lengths.copy_(
            torch.tensor(graph_topology, dtype=torch.int32, device="cpu")
        )
        self.static_cu_seqlens_cpu.copy_(self._make_cu_seqlens(graph_topology))
        self.static_cu_seqlens.copy_(self.static_cu_seqlens_cpu, non_blocking=True)

        address_stable = (
            self.static_sequence_lengths.data_ptr() == self._host_topology_address
        )
        self._print_once(
            "host_topology_updated",
            f"host topology updated: address_stable={address_stable}",
        )

    def _prepare_static_max_seqlen(
        self,
        max_seqlen: torch.Tensor | None,
        graph_topology: tuple[int, ...],
    ) -> None:
        if max_seqlen is None:
            return
        max_value = max(graph_topology)
        if self.static_max_seqlen is None:
            self.static_max_seqlen = max_seqlen.clone()
            self.static_max_seqlen_cpu = torch.tensor(
                max_value,
                dtype=max_seqlen.dtype,
                device="cpu",
            )
        assert self.static_max_seqlen_cpu is not None
        self.static_max_seqlen_cpu.fill_(max_value)
        self.static_max_seqlen.copy_(self.static_max_seqlen_cpu, non_blocking=True)

    def _capture(
        self,
        hidden_states: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        graph_topology: tuple[int, ...],
        num_audios: int,
    ) -> torch.Tensor:
        self._prepare_static_inputs(hidden_states, graph_topology)
        self._prepare_static_max_seqlen(max_seqlen, graph_topology)
        assert self.static_input is not None
        assert self.static_cu_seqlens is not None
        assert self.static_sequence_lengths is not None

        self._print_once("warmup", "warmup begin")
        for _ in range(2):
            self._run_eager(
                self.static_input,
                self.static_cu_seqlens,
                self.static_max_seqlen,
                self.static_sequence_lengths,
                num_audios,
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
                num_audios,
            )
        self._print_once("capture_complete", "capture complete")
        return self.static_output[: hidden_states.shape[0]].clone()

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
        num_audios: int,
    ) -> torch.Tensor:
        graph_topology = self._check_eligibility(
            hidden_states, sequence_lengths, num_audios
        )
        if graph_topology is None:
            return self._run_eager(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
                num_audios,
            )

        actual_tokens = hidden_states.shape[0]
        if self.graph is None:
            try:
                return self._capture(
                    hidden_states,
                    max_seqlen,
                    graph_topology,
                    num_audios,
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

        assert self.static_output is not None
        self._prepare_static_inputs(hidden_states, graph_topology)
        self._prepare_static_max_seqlen(max_seqlen, graph_topology)
        self._print_once(
            "replay_begin",
            f"replay begin: actual={actual_tokens}, padded={self.num_tokens}",
        )
        self.graph.replay()
        self._print_once(
            "replay_complete",
            f"replay complete: returned={actual_tokens}",
        )
        return self.static_output[:actual_tokens].clone()

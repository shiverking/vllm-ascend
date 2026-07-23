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
from dataclasses import dataclass
from itertools import accumulate
from threading import Lock
from typing import Any

import torch
from tqdm import tqdm
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.platforms import current_platform


AUDIO_ENCODER_PROMPT_ATTENTION_ALIGNMENT = 128
_PROMPT_ATTENTION_MASK: ContextVar[torch.Tensor | None] = ContextVar(
    "audio_encoder_prompt_attention_mask",
    default=None,
)


def get_audio_encoder_prompt_attention_mask() -> torch.Tensor | None:
    """Return the fixed Device mask while running an audio encoder graph."""

    return _PROMPT_ATTENTION_MASK.get()


@contextmanager
def use_audio_encoder_prompt_attention(
    mask: torch.Tensor,
) -> Iterator[None]:
    """Select PromptFlashAttention for one encoder-body graph."""

    token = _PROMPT_ATTENTION_MASK.set(mask)
    try:
        yield
    finally:
        _PROMPT_ATTENTION_MASK.reset(token)


def align_audio_encoder_prompt_attention_tokens(num_tokens: int) -> int:
    """Align the PromptFlashAttention sequence while keeping its body small."""

    if num_tokens <= 0:
        raise ValueError("audio encoder graph size must be positive")
    alignment = AUDIO_ENCODER_PROMPT_ATTENTION_ALIGNMENT
    return ((num_tokens + alignment - 1) // alignment) * alignment


def build_padded_attention_mask(
    sequence_lengths: Sequence[int],
    graph_size: int = AUDIO_ENCODER_PROMPT_ATTENTION_ALIGNMENT,
) -> tuple[torch.Tensor, int]:
    """Build a block mask and isolate padding as an independent sequence."""

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
    mask = torch.ones(
        (graph_size, graph_size),
        dtype=torch.bool,
        device="cpu",
    )
    offset = 0
    for length in graph_topology:
        end = offset + length
        mask[offset:end, offset:end] = False
        offset = end
    return mask.contiguous(), dummy_tokens


@dataclass(frozen=True)
class AudioExecutionChunk:
    runner: FixedAudioEncoderAclGraphRunner | None
    sequence_start: int
    sequence_end: int
    token_start: int
    token_end: int

    @property
    def actual_tokens(self) -> int:
        return self.token_end - self.token_start


class FixedAudioEncoderAclGraphRunner:
    """One fixed-size Qwen3-ASR encoder graph with a dynamic Device mask."""

    def __init__(
        self,
        encoder: Any,
        eager_forward: Callable[..., torch.Tensor],
        num_tokens: int,
    ) -> None:
        if num_tokens <= 0:
            raise ValueError("audio encoder ACLGraph sizes must be positive")
        self.encoder = encoder
        self.eager_forward = eager_forward
        self.num_tokens = num_tokens
        self.attention_tokens = align_audio_encoder_prompt_attention_tokens(
            num_tokens
        )
        self.graph: Any | None = None
        self.static_input: torch.Tensor | None = None
        self.static_cu_seqlens: torch.Tensor | None = None
        self.static_max_seqlen: torch.Tensor | None = None
        self.static_sequence_lengths: torch.Tensor | None = None
        self.static_attention_mask: torch.Tensor | None = None
        self.host_attention_mask: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None

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
        capture_mask, _ = build_padded_attention_mask(
            (self.num_tokens,),
            self.attention_tokens,
        )
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
    ) -> None:
        if self.graph is not None:
            return
        try:
            self._capture(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
            )
            torch.npu.synchronize()
        except Exception:
            self.graph = None
            self.static_input = None
            self.static_cu_seqlens = None
            self.static_max_seqlen = None
            self.static_sequence_lengths = None
            self.static_attention_mask = None
            self.host_attention_mask = None
            self.static_output = None
            raise

    def replay(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        if self.graph is None:
            raise RuntimeError("audio encoder graph was not captured")
        assert self.static_input is not None
        assert self.static_attention_mask is not None
        assert self.host_attention_mask is not None
        assert self.static_output is not None
        topology = tuple(sequence_lengths.tolist())
        actual_tokens = hidden_states.shape[0]
        if (
            not topology
            or any(length <= 0 for length in topology)
            or sum(topology) != actual_tokens
            or actual_tokens > self.num_tokens
        ):
            raise ValueError(
                "invalid audio graph chunk: "
                f"tokens={actual_tokens}, graph={self.num_tokens}, "
                f"seq_lens={list(topology)}"
            )
        body_padding = self.num_tokens - actual_tokens
        mask, attention_padding = build_padded_attention_mask(
            topology,
            self.attention_tokens,
        )
        self.static_input[:actual_tokens].copy_(hidden_states)
        if body_padding:
            self.static_input[actual_tokens:].zero_()
        self.host_attention_mask.copy_(mask)
        self.static_attention_mask.copy_(
            self.host_attention_mask,
            non_blocking=True,
        )
        self.graph.replay()
        return (
            self.static_output[:actual_tokens].clone(),
            body_padding,
            attention_padding,
        )

class AudioEncoderAclGraphPool:
    """Run nearest-size encoder graphs without splitting attention sequences."""

    def __init__(
        self,
        encoder: Any,
        eager_forward: Callable[..., torch.Tensor],
        graph_sizes: Sequence[int],
    ) -> None:
        sizes = tuple(sorted(set(graph_sizes)))
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("audio encoder ACLGraph sizes must be positive")
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
        self._batch_count = 0
        self._execution_lock = Lock()
        self._execution_stream: Any | None = None
        self._input_ready_event: Any | None = None
        self._output_ready_event: Any | None = None
        logger.info(
            "[310P_AUDIO_GRAPH] pool enabled: sizes=%s",
            list(sizes),
        )

    def _capture_runner(self, runner: FixedAudioEncoderAclGraphRunner) -> None:
        hidden_size = int(self.encoder.ln_post.normalized_shape[0])
        hidden_states = torch.zeros(
            (runner.num_tokens, hidden_size),
            dtype=self.encoder.dtype,
            device=self.encoder.device,
        )
        sequence_lengths = torch.tensor(
            [runner.num_tokens],
            dtype=torch.int32,
            device="cpu",
        ).contiguous()
        cu_seqlens = self._make_cu_seqlens((runner.num_tokens,)).to(
            self.encoder.device
        )
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

        self._execution_stream = torch.npu.Stream()
        self._input_ready_event = torch.npu.Event()
        self._output_ready_event = torch.npu.Event()
        captured: list[int] = []
        with torch.npu.stream(self._execution_stream):
            with tqdm(
                self.graph_sizes,
                desc="Capturing audio encoder graphs",
                unit="graph",
                disable=not show_progress,
            ) as progress:
                for size in progress:
                    if show_progress:
                        progress.set_postfix(tokens=size)
                    runner = self.runners[size]
                    try:
                        self._capture_runner(runner)
                    except Exception as exc:
                        raise RuntimeError(
                            "Audio encoder ACLGraph capture failed for "
                            f"size {size}: {type(exc).__name__}: {exc}"
                        ) from exc
                    captured.append(size)
        torch.npu.synchronize()
        logger.info(
            "[310P_AUDIO_GRAPH] capture complete: sizes=%s",
            captured,
        )
        return tuple(captured)

    @staticmethod
    def _make_cu_seqlens(topology: Sequence[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *accumulate(topology)],
            dtype=torch.int32,
            device="cpu",
        ).contiguous()

    def _nearest_runner(
        self,
        actual_tokens: int,
    ) -> FixedAudioEncoderAclGraphRunner | None:
        for size in self.graph_sizes:
            if size >= actual_tokens:
                return self.runners[size]
        return None

    def _build_plan(
        self,
        topology: Sequence[int],
    ) -> tuple[AudioExecutionChunk, ...]:
        topology = tuple(int(length) for length in topology)
        if not topology or any(length <= 0 for length in topology):
            raise ValueError("audio encoder topology must be positive")

        offsets = tuple(accumulate((0, *topology)))
        count = len(topology)
        plan_costs: list[tuple[int, int, int] | None] = [None] * (
            count + 1
        )
        plans: list[tuple[AudioExecutionChunk, ...] | None] = [None] * (
            count + 1
        )
        plan_costs[count] = (0, 0, 0)
        plans[count] = ()

        for start in range(count - 1, -1, -1):
            candidates: list[
                tuple[
                    tuple[int, int, int],
                    tuple[AudioExecutionChunk, ...],
                ]
            ] = []
            if topology[start] > self.graph_sizes[-1]:
                tail_cost = plan_costs[start + 1]
                tail_plan = plans[start + 1]
                assert tail_cost is not None and tail_plan is not None
                eager_chunk = AudioExecutionChunk(
                    runner=None,
                    sequence_start=start,
                    sequence_end=start + 1,
                    token_start=offsets[start],
                    token_end=offsets[start + 1],
                )
                candidates.append(
                    (
                        (
                            tail_cost[0] + topology[start],
                            tail_cost[1],
                            tail_cost[2] + 1,
                        ),
                        (eager_chunk, *tail_plan),
                    )
                )
            else:
                graph_options: list[
                    tuple[int, int, FixedAudioEncoderAclGraphRunner]
                ] = []
                actual_tokens = 0
                for end in range(start + 1, count + 1):
                    actual_tokens += topology[end - 1]
                    runner = self._nearest_runner(actual_tokens)
                    if runner is None:
                        break
                    graph_options.append((end, actual_tokens, runner))

                # Prefer a larger first chunk when all aggregate costs tie.
                for end, actual_tokens, runner in reversed(graph_options):
                    tail_cost = plan_costs[end]
                    tail_plan = plans[end]
                    assert tail_cost is not None and tail_plan is not None
                    padding = runner.num_tokens - actual_tokens
                    graph_chunk = AudioExecutionChunk(
                        runner=runner,
                        sequence_start=start,
                        sequence_end=end,
                        token_start=offsets[start],
                        token_end=offsets[end],
                    )
                    candidates.append(
                        (
                            (
                                tail_cost[0],
                                tail_cost[1] + padding,
                                tail_cost[2] + 1,
                            ),
                            (graph_chunk, *tail_plan),
                        )
                    )

            if not candidates:
                raise RuntimeError(
                    f"failed to build audio graph plan at sequence {start}"
                )
            best_cost, best_plan = min(candidates, key=lambda item: item[0])
            plan_costs[start] = best_cost
            plans[start] = best_plan

        result = plans[0]
        assert result is not None
        return result

    def _check_common_eligibility(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> str | None:
        if self.encoder.enforce_eager:
            return "enforce_eager"
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
        topology = tuple(sequence_lengths.tolist())
        if (
            not topology
            or any(length <= 0 for length in topology)
            or sum(topology) != hidden_states.shape[0]
        ):
            return "invalid_topology"
        if any(runner.graph is None for runner in self.runners.values()):
            return "graph_not_captured"
        return None

    def _run_eager_chunk(
        self,
        chunk: AudioExecutionChunk,
        hidden_states: torch.Tensor,
        topology: tuple[int, ...],
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        chunk_topology = topology[
            chunk.sequence_start : chunk.sequence_end
        ]
        chunk_sequence_lengths = torch.tensor(
            chunk_topology,
            dtype=torch.int32,
            device="cpu",
        ).contiguous()
        chunk_cu_seqlens = self._make_cu_seqlens(chunk_topology).to(
            hidden_states.device
        )
        chunk_max_seqlen = None
        if max_seqlen is not None:
            chunk_max_seqlen = self.encoder.compute_attn_mask_seqlen(
                chunk_cu_seqlens
            )
        return self.eager_forward(
            self.encoder,
            hidden_states[chunk.token_start : chunk.token_end],
            chunk_cu_seqlens,
            chunk_max_seqlen,
            chunk_sequence_lengths,
        )

    def _execute_plan(
        self,
        plan: tuple[AudioExecutionChunk, ...],
        hidden_states: torch.Tensor,
        topology: tuple[int, ...],
        max_seqlen: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[int], int, int, int]:
        outputs: list[torch.Tensor] = []
        graph_sizes: list[int] = []
        body_padding_tokens = 0
        attention_padding_tokens = 0
        eager_tokens = 0
        for chunk in plan:
            if chunk.runner is None:
                outputs.append(
                    self._run_eager_chunk(
                        chunk,
                        hidden_states,
                        topology,
                        max_seqlen,
                    )
                )
                eager_tokens += chunk.actual_tokens
                continue

            chunk_topology = topology[
                chunk.sequence_start : chunk.sequence_end
            ]
            chunk_sequence_lengths = torch.tensor(
                chunk_topology,
                dtype=torch.int32,
                device="cpu",
            ).contiguous()
            output, body_padding, attention_padding = chunk.runner.replay(
                hidden_states[chunk.token_start : chunk.token_end],
                chunk_sequence_lengths,
            )
            outputs.append(output)
            graph_sizes.append(chunk.runner.num_tokens)
            body_padding_tokens += body_padding
            attention_padding_tokens += attention_padding

        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return (
            output,
            graph_sizes,
            body_padding_tokens,
            attention_padding_tokens,
            eager_tokens,
        )

    def _execute_plan_stream_safe(
        self,
        plan: tuple[AudioExecutionChunk, ...],
        hidden_states: torch.Tensor,
        topology: tuple[int, ...],
        max_seqlen: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[int], int, int, int]:
        assert self._execution_stream is not None
        assert self._input_ready_event is not None
        assert self._output_ready_event is not None
        with self._execution_lock:
            caller_stream = torch.npu.current_stream()
            self._input_ready_event.record(caller_stream)
            self._execution_stream.wait_event(self._input_ready_event)
            with torch.npu.stream(self._execution_stream):
                result = self._execute_plan(
                    plan,
                    hidden_states,
                    topology,
                    max_seqlen,
                )
                self._output_ready_event.record(self._execution_stream)
            caller_stream.wait_event(self._output_ready_event)
        return result

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        self._batch_count += 1
        batch_id = self._batch_count
        reason = self._check_common_eligibility(
            hidden_states,
            sequence_lengths,
        )
        if reason is not None:
            logger.info(
                "[310P_AUDIO_GRAPH] batch=%d, hit=False, actual=%d, "
                "graphs=[], body_padding=0, attention_padding=0, "
                "eager=%d, reason=%s",
                batch_id,
                hidden_states.shape[0],
                hidden_states.shape[0],
                reason,
            )
            return self.eager_forward(
                self.encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
            )

        topology = tuple(sequence_lengths.tolist())
        plan = self._build_plan(topology)
        (
            output,
            graph_sizes,
            body_padding_tokens,
            attention_padding_tokens,
            eager_tokens,
        ) = (
            self._execute_plan_stream_safe(
                plan,
                hidden_states,
                topology,
                max_seqlen,
            )
        )
        hit: bool | str
        if graph_sizes and eager_tokens:
            hit = "partial"
        else:
            hit = bool(graph_sizes)
        logger.info(
            "[310P_AUDIO_GRAPH] batch=%d, hit=%s, actual=%d, graphs=%s, "
            "body_padding=%d, attention_padding=%d, eager=%d",
            batch_id,
            hit,
            hidden_states.shape[0],
            graph_sizes,
            body_padding_tokens,
            attention_padding_tokens,
            eager_tokens,
        )
        return output

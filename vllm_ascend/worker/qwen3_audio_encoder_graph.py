"""Fixed-shape masked ACLGraphs for the Qwen3-ASR audio encoder body.

The convolutional frontend and final per-audio split remain outside the graph.
Each graph uses a device attention mask, so replay does not update FIA tasks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate
from threading import Lock
from typing import Any, Iterator, Sequence

import torch
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.worker.encoder_acl_graph import (
    build_audio_attention_mask,
    set_encoder_forward_context,
)


@dataclass(frozen=True)
class AudioGraphChunk:
    sequence_start: int
    sequence_end: int
    token_start: int
    token_end: int
    budget: int | None

    @property
    def actual_tokens(self) -> int:
        return self.token_end - self.token_start


def plan_audio_graph_chunks(
    topology: Sequence[int], budgets: Sequence[int]
) -> tuple[AudioGraphChunk, ...]:
    """Minimize eager tokens, then padding, then the number of replays."""
    lengths = tuple(int(length) for length in topology)
    sizes = tuple(sorted(set(int(size) for size in budgets)))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("audio encoder sequence lengths must be positive")
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("audio encoder graph budgets must be positive")

    offsets = tuple(accumulate((0, *lengths)))
    costs: list[tuple[int, int, int] | None] = [None] * (len(lengths) + 1)
    plans: list[tuple[AudioGraphChunk, ...] | None] = [None] * (
        len(lengths) + 1
    )
    costs[-1] = (0, 0, 0)
    plans[-1] = ()
    for start in range(len(lengths) - 1, -1, -1):
        candidates = []
        actual = 0
        for end in range(start + 1, len(lengths) + 1):
            actual += lengths[end - 1]
            budget = next((size for size in sizes if size >= actual), None)
            if budget is None:
                break
            tail_cost = costs[end]
            tail_plan = plans[end]
            assert tail_cost is not None and tail_plan is not None
            chunk = AudioGraphChunk(start, end, offsets[start], offsets[end], budget)
            candidates.append(
                (
                    (tail_cost[0], tail_cost[1] + budget - actual, tail_cost[2] + 1),
                    (chunk, *tail_plan),
                )
            )

        if lengths[start] > sizes[-1]:
            tail_cost = costs[start + 1]
            tail_plan = plans[start + 1]
            assert tail_cost is not None and tail_plan is not None
            chunk = AudioGraphChunk(
                start, start + 1, offsets[start], offsets[start + 1], None
            )
            candidates.append(
                (
                    (tail_cost[0] + lengths[start], tail_cost[1], tail_cost[2] + 1),
                    (chunk, *tail_plan),
                )
            )

        if not candidates:
            raise RuntimeError(f"no audio encoder graph plan at sequence {start}")
        costs[start], plans[start] = min(
            candidates,
            key=lambda item: (item[0], -item[1][0].actual_tokens),
        )

    result = plans[0]
    assert result is not None
    return result


class FixedAudioEncoderGraph:
    """One graph with fixed input/output addresses and a mutable device mask."""

    def __init__(self, model: Any, budget: int, device: torch.device):
        self.model = model
        self.budget = budget
        self.device = device
        self.graph: Any | None = None
        self.hidden_states: torch.Tensor | None = None
        self.cu_seqlens: torch.Tensor | None = None
        self.sequence_lengths: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.host_mask: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def _forward(self) -> torch.Tensor:
        assert self.hidden_states is not None
        assert self.cu_seqlens is not None
        assert self.sequence_lengths is not None
        assert self.mask is not None
        tower = self.model.audio_tower
        with set_encoder_forward_context(
            self.budget, False, attention_mask=self.mask
        ):
            return tower._forward_encoder_body(
                self.hidden_states,
                self.cu_seqlens,
                tower.compute_attn_mask_seqlen(self.cu_seqlens),
                self.sequence_lengths,
            )

    def capture(self, graph_pool: Any) -> None:
        hidden_size = int(self.model.audio_tower.ln_post.normalized_shape[0])
        self.hidden_states = torch.zeros(
            (self.budget, hidden_size), device=self.device, dtype=torch.float16
        )
        self.cu_seqlens = torch.tensor(
            [0, self.budget], device=self.device, dtype=torch.int32
        )
        self.sequence_lengths = torch.tensor(
            [self.budget], device="cpu", dtype=torch.int32
        )
        host_mask = build_audio_attention_mask(self.sequence_lengths, self.budget)
        self.host_mask = torch.empty_like(host_mask)
        self.host_mask.copy_(host_mask)
        self.mask = host_mask.to(self.device)

        with torch.inference_mode():
            for _ in range(2):
                self._forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=graph_pool):
                output = self._forward()
                self.output = torch.empty_like(output)
                self.output.copy_(output)
            self.graph = graph
            torch.npu.synchronize()

    def replay(
        self, hidden_states: torch.Tensor, topology: Sequence[int]
    ) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError(f"audio graph {self.budget} was not captured")
        assert self.hidden_states is not None
        assert self.mask is not None
        assert self.host_mask is not None
        assert self.output is not None
        actual = hidden_states.shape[0]
        if sum(topology) != actual or actual > self.budget:
            raise ValueError("audio graph input and sequence lengths do not match")

        self.hidden_states[:actual].copy_(hidden_states)
        if actual < self.budget:
            self.hidden_states[actual:].zero_()
        self.host_mask.copy_(
            build_audio_attention_mask(topology, self.budget)
        )
        self.mask.copy_(self.host_mask)
        self.graph.replay()
        return self.output[:actual].clone()


class Qwen3AudioEncoderGraphPool:
    """Capture basis graphs at startup and replay complete attention sequences."""

    def __init__(self, model: Any, device: torch.device, budgets: Sequence[int]):
        self.model = model
        self.device = device
        self.budgets = tuple(sorted(set(budgets)))
        window = model.audio_tower.get_attention_window_tokens()
        expected = (window // 4, window, 2 * window, 4 * window)
        if window % 4 or self.budgets != expected:
            raise ValueError(
                "Qwen3-ASR audio graph budgets must be "
                f"{list(expected)} for attention window {window}; "
                f"got {list(self.budgets)}"
            )
        self.runners = {
            budget: FixedAudioEncoderGraph(model, budget, device)
            for budget in self.budgets
        }
        self._lock = Lock()
        self._stream: Any | None = None
        self._input_ready: Any | None = None
        self._output_ready: Any | None = None
        self._calls = 0
        logger.info(
            "[AUDIO_ENCODER_GRAPH] enabled: sizes=%s window=%d",
            list(self.budgets),
            window,
        )

    def capture(self, graph_pool: Any | None = None) -> None:
        pool = graph_pool if graph_pool is not None else current_platform.get_global_graph_pool()
        self._stream = torch.npu.Stream()
        self._input_ready = torch.npu.Event()
        self._output_ready = torch.npu.Event()
        with torch.npu.stream(self._stream):
            for budget in self.budgets:
                try:
                    self.runners[budget].capture(pool)
                except Exception as exc:
                    raise RuntimeError(
                        f"Qwen3-ASR audio graph capture failed at {budget} tokens"
                    ) from exc
        logger.info("[AUDIO_ENCODER_GRAPH] captured sizes=%s", list(self.budgets))

    def _execute_plan(
        self,
        hidden_states: torch.Tensor,
        topology: tuple[int, ...],
        plan: tuple[AudioGraphChunk, ...],
    ) -> tuple[torch.Tensor, list[int], int, int]:
        outputs: list[torch.Tensor] = []
        graph_chunks: list[int] = []
        padding = 0
        eager_tokens = 0
        for chunk in plan:
            chunk_topology = topology[chunk.sequence_start : chunk.sequence_end]
            chunk_input = hidden_states[chunk.token_start : chunk.token_end]
            if chunk.budget is None:
                seq_cpu = torch.tensor(chunk_topology, dtype=torch.int32)
                cu_cpu = torch.tensor(
                    [0, *accumulate(chunk_topology)], dtype=torch.int32
                )
                cu = cu_cpu.to(self.device)
                tower = self.model.audio_tower
                outputs.append(
                    tower._forward_encoder_body(
                        chunk_input,
                        cu,
                        tower.compute_attn_mask_seqlen(cu),
                        seq_cpu,
                    )
                )
                eager_tokens += chunk.actual_tokens
            else:
                outputs.append(
                    self.runners[chunk.budget].replay(chunk_input, chunk_topology)
                )
                graph_chunks.append(chunk.budget)
                padding += chunk.budget - chunk.actual_tokens
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return output, graph_chunks, padding, eager_tokens

    @contextmanager
    def _ordered_stream(self) -> Iterator[None]:
        assert self._stream is not None
        assert self._input_ready is not None
        assert self._output_ready is not None
        with self._lock:
            caller = torch.npu.current_stream()
            self._input_ready.record(caller)
            self._stream.wait_event(self._input_ready)
            with torch.npu.stream(self._stream):
                yield
                self._output_ready.record(self._stream)
            caller.wait_event(self._output_ready)

    @torch.inference_mode()
    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        self._calls += 1
        hidden_states, _, _, sequence_lengths = (
            self.model._prepare_audio_encoder_graph_inputs(mm_kwargs)
        )
        topology = tuple(int(n) for n in sequence_lengths.tolist())
        if (
            hidden_states.dtype != torch.float16
            or hidden_states.device.type != "npu"
            or not hidden_states.is_contiguous()
            or sequence_lengths.device.type != "cpu"
            or sequence_lengths.dtype != torch.int32
            or sum(topology) != hidden_states.shape[0]
        ):
            raise ValueError("unsupported Qwen3-ASR audio graph input")
        plan = plan_audio_graph_chunks(topology, self.budgets)
        with self._ordered_stream():
            output, graph_chunks, padding, eager_tokens = self._execute_plan(
                hidden_states, topology, plan
            )

        item_lengths = [
            spec.output_tokens
            for spec in self.model.get_encoder_cudagraph_item_specs(mm_kwargs)
        ]
        if sum(item_lengths) != output.shape[0]:
            raise ValueError(
                "audio graph outputs do not match per-item token lengths: "
                f"items={item_lengths}, output={output.shape[0]}"
            )
        logger.info(
            "[AUDIO_ENCODER_GRAPH] call=%d items=%d item_tokens=%s "
            "actual=%d seq_lens=%s graphs=%s replays=%d "
            "padding=%d eager=%d graph_used=%s graph_hit=%s state=submitted",
            self._calls,
            len(item_lengths),
            item_lengths,
            hidden_states.shape[0],
            list(topology),
            graph_chunks,
            len(graph_chunks),
            padding,
            eager_tokens,
            bool(graph_chunks),
            bool(graph_chunks) and eager_tokens == 0,
        )
        return list(output.split(item_lengths, dim=0))

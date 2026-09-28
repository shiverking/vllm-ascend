#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""NPU-specific encoder ACL graph: params, runtime context, FIA replay updates, and manager."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch_npu
from vllm.logger import logger
from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata, EncoderCudaGraphManager

from vllm_ascend.utils import vllm_version_is, weak_ref_tensors


SHORT_AUDIO_MASK_GRAPH_MAX_TOKENS = 26
SHORT_AUDIO_MASK_ALIGNMENT = 128

# ---------------------------------------------------------------------------
# Per–encoder-budget ACL graph bookkeeping (ViT FIA tasks)
# ---------------------------------------------------------------------------


@dataclass
class EncoderGraphParams:
    """Mirrors :class:`vllm_ascend.compilation.acl_graph.GraphParams` but keyed by encoder token budget."""

    # TODO: Fully support upstream dual-path encoder graph on Ascend. The
    # current FIA bookkeeping is keyed only by token_budget; dual-path models
    # need graph params separated by (path, token_budget).
    events: dict[int, list[torch.npu.ExternalEvent]] = field(default_factory=dict)
    workspaces: dict[int, torch.Tensor | None] = field(default_factory=dict)
    handles: dict[int, list[Any]] = field(default_factory=dict)
    # Flattened per-forward insertion order (one entry per ViT block invocation).
    attn_params: dict[int, list[tuple]] = field(default_factory=dict)


_encoder_graph_params: EncoderGraphParams | None = None


def set_encoder_graph_params(token_budgets: list[int]) -> None:
    global _encoder_graph_params
    budgets_sorted_unique = sorted(token_budgets)
    _encoder_graph_params = EncoderGraphParams(
        events={b: [] for b in budgets_sorted_unique},
        workspaces={b: None for b in budgets_sorted_unique},
        handles={b: [] for b in budgets_sorted_unique},
        attn_params={b: [] for b in budgets_sorted_unique},
    )


def get_encoder_graph_params() -> EncoderGraphParams | None:
    return _encoder_graph_params


def update_encoder_graph_workspace(token_budget: int, workspace: torch.Tensor) -> None:
    if _encoder_graph_params is None:
        return
    _encoder_graph_params.workspaces[token_budget] = workspace


# ---------------------------------------------------------------------------
# Capture / replay runtime state (thread-local module singleton)
# ---------------------------------------------------------------------------


@dataclass
class EncoderForwardContext:
    """Vision encoder NPUGraph runtime flags and host-side FIA arguments.

    Captured tensors stay on device; FIA ``graph_task_update`` needs Python ``list[int]``
    lengths that are refreshed each replay from encoder metadata buffers on device (see RFC).
    """

    token_budget: int | None = None
    capturing: bool = False
    cu_seqlens_cpu: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


_context = EncoderForwardContext()


def get_encoder_forward_context() -> EncoderForwardContext:
    return _context


def _reset_encoder_forward_context() -> None:
    """Clear replay-time host length fields."""

    _context.token_budget = None
    _context.capturing = False
    _context.cu_seqlens_cpu = None
    _context.attention_mask = None


@contextmanager
def set_encoder_forward_context(
    token_budget: int,
    capturing: bool,
    *,
    cu_seqlens_cpu: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
):
    """Enter encoder graph replay (FIA host args): callers must pass lengths each time.

    On exit, replay host fields are **cleared** (not restored). Tensors must not be reused
    across replays without repopulating from the current batch buffers.
    """

    _context.token_budget = token_budget
    _context.capturing = capturing
    _context.cu_seqlens_cpu = cu_seqlens_cpu
    _context.attention_mask = attention_mask
    try:
        yield _context
    finally:
        _reset_encoder_forward_context()


# ---------------------------------------------------------------------------
# FIA actual_seq_lengths (cu_seqlens -> list[int])
# ---------------------------------------------------------------------------


def maybe_compute_actual_seq_lengths(
    cu_seqlens: torch.Tensor,
    num_query_tokens: int,
    num_kv_tokens: int,
    *,
    cudagraph_mm_encoder: bool = False,
) -> tuple[list[int], list[int]]:
    """Convert ``cu_seqlens`` to FIA host ``actual_seq_lengths``.

    Drops the leading-zero marker; with ``cudagraph_mm_encoder`` filters endpoints and
    aligns the terminal to ``num_query_tokens``; when Q≠KV, scales Q endpoints by
    ``num_kv_tokens // num_query_tokens`` for the KV list.
    """
    flat = cu_seqlens.detach().cpu().view(-1).tolist()
    actual = flat[1:] if flat else flat

    if not cudagraph_mm_encoder:
        pass
    elif num_query_tokens <= 0:
        actual = [0]
    else:
        filtered: list[int] = []
        for end in actual:
            if end <= 0:
                continue
            if end > num_query_tokens:
                break
            if not filtered or end > filtered[-1]:
                filtered.append(end)

        if not filtered or filtered[-1] != num_query_tokens:
            filtered.append(num_query_tokens)
        actual = filtered

    if num_kv_tokens == num_query_tokens:
        return actual, actual

    assert num_kv_tokens % num_query_tokens == 0
    ratio = num_kv_tokens // num_query_tokens
    return actual, [end * ratio for end in actual]


def prepare_padded_sequence_lengths(
    sequence_lengths: torch.Tensor,
    actual_tokens: int,
    token_budget: int,
) -> torch.Tensor:
    sequence_lengths = sequence_lengths.to(
        device="cpu", dtype=torch.int32
    ).contiguous()
    if int(sequence_lengths.sum()) != actual_tokens:
        raise ValueError(
            "Encoder graph sequence lengths do not match its input: "
            f"tokens={actual_tokens}, seq_lens={sequence_lengths.tolist()}"
        )
    padding = token_budget - actual_tokens
    if padding < 0:
        raise ValueError(
            "Encoder graph input exceeds its token budget: "
            f"actual={actual_tokens}, budget={token_budget}"
        )
    if padding:
        sequence_lengths = torch.cat(
            (
                sequence_lengths,
                torch.tensor([padding], dtype=torch.int32),
            )
        )
    return sequence_lengths


def build_short_audio_attention_mask(
    sequence_lengths: torch.Tensor, token_budget: int
) -> torch.Tensor:
    """Keep each real sequence and all padding in separate attention blocks."""
    if token_budget <= 0 or token_budget > SHORT_AUDIO_MASK_GRAPH_MAX_TOKENS:
        raise ValueError(f"Unsupported short audio graph budget: {token_budget}")
    if (
        sequence_lengths.ndim != 1
        or sequence_lengths.numel() == 0
        or sequence_lengths.device.type != "cpu"
        or bool((sequence_lengths <= 0).any())
    ):
        raise ValueError("Short audio graph requires positive CPU sequence lengths")
    actual_tokens = int(sequence_lengths.sum())
    topology = prepare_padded_sequence_lengths(
        sequence_lengths, actual_tokens, token_budget
    ).tolist()
    aligned_tokens = (
        (token_budget + SHORT_AUDIO_MASK_ALIGNMENT - 1)
        // SHORT_AUDIO_MASK_ALIGNMENT
        * SHORT_AUDIO_MASK_ALIGNMENT
    )
    if aligned_tokens > token_budget:
        topology.append(aligned_tokens - token_budget)
    mask = torch.ones((aligned_tokens, aligned_tokens), dtype=torch.bool)
    offset = 0
    for length in topology:
        end = offset + length
        mask[offset:end, offset:end] = False
        offset = end
    return mask.contiguous()


def update_encoder_graph_params(
    update_stream: torch.npu.Stream,
    token_budget: int,
) -> None:
    """Re-bind fused infer attention host tensors inside the encoder NPUGraph (parallel to LLM path).

    This deliberately bypasses :class:`AttentionBackend` — ViT attention is not registered there — but reuses
    the same ``graph_task_update_{begin,end}`` + ``ExternalEvent`` ordering pattern as
    :meth:`AscendAttentionBackendImpl.update_graph_params`.
    """

    params = get_encoder_graph_params()
    if params is None or token_budget not in params.handles:
        return

    handles = params.handles[token_budget]
    events = params.events[token_budget]
    attn_blocks = params.attn_params[token_budget]
    workspace = params.workspaces.get(token_budget)

    if len(handles) != len(events) or len(handles) != len(attn_blocks):
        raise RuntimeError(
            "Encoder graph bookkeeping is inconsistent: "
            f"budget={token_budget} handles={len(handles)} "
            f"events={len(events)} attn_blocks={len(attn_blocks)}"
        )

    with torch.npu.stream(update_stream):
        for handle, event, packed in zip(handles, events, attn_blocks):
            (
                query,
                key,
                value,
                block_table,
                attn_mask,
                block_size,
                num_kv_heads,
                num_heads,
                scale,
                output,
                softmax_lse,
            ) = packed

            num_query_tokens = query.shape[0]
            num_kv_tokens = key.shape[0]
            cu_seqlens_cpu = get_encoder_forward_context().cu_seqlens_cpu

            actual_seq_lengths_q, actual_seq_lengths_kv = maybe_compute_actual_seq_lengths(
                cu_seqlens_cpu,
                num_query_tokens,
                num_kv_tokens,
                cudagraph_mm_encoder=True,
            )

            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)


# ---------------------------------------------------------------------------
# Encoder NPUGraph manager
# ---------------------------------------------------------------------------
class EncoderAclGraphManager(EncoderCudaGraphManager):
    """Hooks encoder capture/replay into Ascend FIA graph-task infrastructure."""

    supports_runtime_sequence_lengths = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.graph_pool = current_platform.get_global_graph_pool()
        self.update_stream: torch.npu.Stream | None = None
        self.short_audio_attention_masks: dict[int, torch.Tensor] = {}
        self.short_audio_host_masks: dict[int, torch.Tensor] = {}
        self.diagnostic_sync_stage = (self.vllm_config.additional_config or {}).get(
            "audio_encoder_graph_diag_sync", "off"
        )
        if self.diagnostic_sync_stage not in {"off", "capture", "inputs", "replay"}:
            raise ValueError(
                "audio_encoder_graph_diag_sync must be one of off, capture, "
                "inputs, replay"
            )
        if self.diagnostic_sync_stage != "off":
            logger.warning(
                "[ENCODER_GRAPH_DIAG] enabled: sync_stage=%s; diagnostic run only",
                self.diagnostic_sync_stage,
            )

    def _diagnostic_sync(self, stage: str, token_budget: int) -> None:
        if self.diagnostic_sync_stage != stage:
            return
        logger.warning(
            "[ENCODER_GRAPH_DIAG] sync begin: stage=%s, budget=%d",
            stage,
            token_budget,
        )
        try:
            torch.npu.synchronize()
        except Exception:
            logger.exception(
                "[ENCODER_GRAPH_DIAG] sync failed: stage=%s, budget=%d",
                stage,
                token_budget,
            )
            raise
        logger.warning(
            "[ENCODER_GRAPH_DIAG] sync complete: stage=%s, budget=%d",
            stage,
            token_budget,
        )

    def _uses_short_audio_mask(self, token_budget: int) -> bool:
        return (
            self.model.__class__.__name__ == "Qwen3ASRForConditionalGeneration"
            and token_budget <= SHORT_AUDIO_MASK_GRAPH_MAX_TOKENS
        )

    def capture(self, graph_pool: Any | None = None):
        encoder_graph_pool = graph_pool if graph_pool is not None else self.graph_pool
        self.graph_pool = encoder_graph_pool

        set_encoder_graph_params(self.token_budgets)

        super().capture(graph_pool=encoder_graph_pool)

        weak_ref_workspaces()

    def _capture_budget_graph(self, token_budget: int, path: str = "default"):
        logger.debug(
            "Capturing encoder aclgraph for budget=%d, max_batch_size=%d, max_frames_per_batch=%d",
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
        )

        if vllm_version_is("0.23.0"):
            capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
                token_budget,
                self.max_batch_size,
                self.max_frames_per_batch,
                self.device,
                self.dtype,
            )
        else:
            capture_inputs = self.model.prepare_encoder_cudagraph_capture_inputs(
                token_budget,
                self.max_batch_size,
                self.max_frames_per_batch,
                self.device,
                self.dtype,
                path,
            )

        values = capture_inputs.values
        attention_mask = None
        host_attention_mask = None
        if self._uses_short_audio_mask(token_budget):
            host_attention_mask = build_short_audio_attention_mask(
                values["sequence_lengths"], token_budget
            )
            attention_mask = host_attention_mask.to(device=self.device)
        with torch.inference_mode():
            with set_encoder_forward_context(
                token_budget, False, attention_mask=attention_mask
            ):
                if vllm_version_is("0.23.0"):
                    output = self.model.encoder_cudagraph_forward(dict(values))
                else:
                    output = self.model.encoder_cudagraph_forward(dict(values), path=path)
            output_buffer = torch.empty_like(output)

        graph = torch.npu.NPUGraph()
        with (
            set_encoder_forward_context(
                token_budget, True, attention_mask=attention_mask
            ),
            torch.inference_mode(),
            torch.npu.graph(graph, self.graph_pool),
        ):
            if vllm_version_is("0.23.0"):
                output = self.model.encoder_cudagraph_forward(dict(values))
            else:
                output = self.model.encoder_cudagraph_forward(dict(values), path=path)
            output_buffer.copy_(output)

        graph_meta = BudgetGraphMetadata(
            token_budget=token_budget,
            max_batch_size=self.max_batch_size,
            max_frames_per_batch=self.max_frames_per_batch,
            graph=graph,
            input_buffers=values,
            output_buffer=weak_ref_tensors(output_buffer),
        )
        if attention_mask is not None:
            self.short_audio_attention_masks[token_budget] = attention_mask
            self.short_audio_host_masks[token_budget] = host_attention_mask
            logger.info(
                "Audio encoder graph captured with device-masked attention: "
                "budget=%d, attention_tokens=%d",
                token_budget,
                attention_mask.shape[0],
            )
        if vllm_version_is("0.23.0"):
            self.budget_graphs[token_budget] = graph_meta
        else:
            graph_set = self._get_graph_set(path)
            graph_set[token_budget] = graph_meta
        if self.diagnostic_sync_stage != "off":
            params = get_encoder_graph_params()
            workspace = None if params is None else params.workspaces.get(token_budget)
            workspace_bytes = (
                workspace.numel() * workspace.element_size()
                if isinstance(workspace, torch.Tensor)
                else 0
            )
            task_count = 0 if params is None else len(params.handles[token_budget])
            logger.warning(
                "[ENCODER_GRAPH_DIAG] capture stored: budget=%d, attention=%s, "
                "sequence_lengths=%s, fia_tasks=%d, workspace_bytes=%d",
                token_budget,
                "masked_prompt" if attention_mask is not None else "fia",
                (
                    None
                    if values.get("sequence_lengths") is None
                    else values["sequence_lengths"].tolist()
                ),
                task_count,
                workspace_bytes,
            )
            self._diagnostic_sync("capture", token_budget)

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
    ) -> torch.Tensor | None:
        num_items = len(self._get_item_specs(mm_kwargs))
        if vllm_version_is("0.23.0"):
            if token_budget not in self.budget_graphs:
                self.graph_misses += num_items
                return None
            graph_meta = self.budget_graphs[token_budget]
        else:
            graph_set = self._get_graph_set(path)
            if token_budget not in graph_set:
                self.graph_misses += num_items
                return None
            graph_meta = graph_set[token_budget]

        if vllm_version_is("0.23.0"):
            replay = self.model.prepare_encoder_cudagraph_replay_buffers(
                mm_kwargs,
                self.max_batch_size,
                self.max_frames_per_batch,
            )
            buffer_items = ((key, graph_meta.input_buffers[key]) for key in self.config.buffer_keys)
        else:
            replay = self.model.prepare_encoder_cudagraph_replay_buffers(
                mm_kwargs,
                self.max_batch_size,
                self.max_frames_per_batch,
                path,
            )
            buffer_items = graph_meta.input_buffers.items()

        for key, buf in buffer_items:
            src = replay.values.get(key)
            if src is None:
                continue
            if src.ndim == 0:
                buf.copy_(src)
            else:
                padding_logic = self.config.padding_logics.get(key, self._copy_padded_buffer)
                padding_logic(buf, src)

        sequence_lengths = replay.values.get("sequence_lengths")
        if sequence_lengths is not None:
            actual_tokens = int(sequence_lengths.sum())
            padded_sequence_lengths = prepare_padded_sequence_lengths(
                sequence_lengths,
                actual_tokens,
                token_budget,
            )
            cu_seqlens_cpu = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32),
                    padded_sequence_lengths.cumsum(
                        0, dtype=torch.int32
                    ),
                )
            )
            logger.debug(
                "Audio encoder ACLGraph replay: actual_tokens=%d, "
                "budget=%d, padding_tokens=%d",
                actual_tokens,
                token_budget,
                token_budget - actual_tokens,
            )
            if self._uses_short_audio_mask(token_budget):
                mask = build_short_audio_attention_mask(
                    sequence_lengths, token_budget
                )
                host_mask = self.short_audio_host_masks[token_budget]
                host_mask.copy_(mask)
                self.short_audio_attention_masks[token_budget].copy_(host_mask)
        else:
            cu_seqlens = graph_meta.input_buffers.get("cu_seqlens")
            cu_seqlens_cpu = (
                None if cu_seqlens is None else cu_seqlens.cpu()
            )

        if self.diagnostic_sync_stage != "off":
            logger.warning(
                "[ENCODER_GRAPH_DIAG] inputs updated: budget=%d, "
                "sequence_lengths=%s, cu_seqlens=%s, attention=%s",
                token_budget,
                None if sequence_lengths is None else sequence_lengths.tolist(),
                None if cu_seqlens_cpu is None else cu_seqlens_cpu.tolist(),
                "masked_prompt" if self._uses_short_audio_mask(token_budget) else "fia",
            )
            self._diagnostic_sync("inputs", token_budget)

        update_stream = self.update_stream
        if update_stream is None:
            update_stream = torch.npu.Stream()

        if self.diagnostic_sync_stage != "off":
            logger.warning(
                "[ENCODER_GRAPH_DIAG] replay begin: budget=%d", token_budget
            )
        graph_meta.graph.replay()

        if not self._uses_short_audio_mask(token_budget):
            if self.diagnostic_sync_stage != "off":
                logger.warning(
                    "[ENCODER_GRAPH_DIAG] FIA update begin: budget=%d", token_budget
                )
            with set_encoder_forward_context(
                token_budget,
                False,
                cu_seqlens_cpu=cu_seqlens_cpu,
            ):
                update_encoder_graph_params(update_stream, token_budget)
            if self.diagnostic_sync_stage != "off":
                logger.warning(
                    "[ENCODER_GRAPH_DIAG] FIA update submitted: budget=%d",
                    token_budget,
                )

        if self.diagnostic_sync_stage != "off":
            self._diagnostic_sync("replay", token_budget)

        self.graph_hits += num_items
        logger.debug(
            "Audio encoder ACLGraph replay submitted: budget=%d, attention=%s",
            token_budget,
            "masked_prompt" if self._uses_short_audio_mask(token_budget) else "fia",
        )
        return graph_meta.output_buffer


def weak_ref_workspaces() -> None:
    params = get_encoder_graph_params()
    if params is None:
        return
    for budget, ws in list(params.workspaces.items()):
        if ws is None:
            continue
        params.workspaces[budget] = weak_ref_tensors(ws)

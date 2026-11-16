# SPDX-License-Identifier: Apache-2.0
"""Fixed-time CNN graphs, partitioned only along independent audio chunks."""

from collections.abc import Sequence
from threading import Lock
from typing import Any

import torch
import torch_npu
from tqdm import tqdm
from vllm.logger import logger


def normalize_frontend_batches(batches: Sequence[int], limit: int) -> tuple[int, ...]:
    if not isinstance(batches, (list, tuple)) or any(type(batch) is not int or batch <= 0 for batch in batches):
        raise ValueError("audio_encoder_frontend_graph_batches must be a list of positive integers")
    sizes = tuple(sorted(set(batches)))
    if sizes and sizes[-1] > limit:
        raise ValueError(
            f"audio_encoder_frontend_graph_batches contains {sizes[-1]}, exceeding "
            f"the model conv_chunksize={limit} memory limit; use smaller batches"
        )
    return sizes


def plan_frontend_batches(actual: int, batches: Sequence[int]) -> list[tuple[int, int]]:
    """Return (real chunks, graph batch); pad only the final batch."""
    plan = []
    while actual > 0:
        real = min(actual, batches[-1])
        budget = next(size for size in batches if size >= real)
        plan.append((real, budget))
        actual -= real
    return plan


class AudioFrontendGraphPool:
    def __init__(self, tower: Any, batches: Sequence[int]):
        self.batches = normalize_frontend_batches(batches, tower.conv_chunksize)
        if not self.batches:
            raise ValueError("audio frontend graph pool requires at least one batch")
        self.forward_eager = tower._forward_cnn_frontend
        self.device = tower.device
        self.time = 2 * tower.n_window
        self.mel_bins = tower.num_mel_bins
        self.runners: dict[int, tuple[Any, torch.Tensor, torch.Tensor]] = {}
        self._lock = Lock()
        self._stream = None
        self._input_ready = None
        self._output_ready = None
        self._calls = 0

    @torch.inference_mode()
    def capture(self) -> None:
        # Conv2D can dispatch to non-capturable ACLOP if either option is on.
        # This scope is startup-only: never change process-wide options during
        # replay or eager fallback. Some TorchNPU versions expose the internal
        # format config as write-only, so read the underlying option (default on).
        previous_jit = not torch.npu.is_jit_compile_false()
        previous_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") != b"disable"
        failed = False
        try:
            torch.npu.synchronize()
            torch.npu.config.allow_internal_format = False
            if previous_jit:
                torch.npu.set_compile_mode(jit_compile=False)
            logger.info(
                "[AUDIO_FRONTEND_GRAPH] capture settings: allow_internal_format=%s jit_compile=%s",
                torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") != b"disable",
                not torch.npu.is_jit_compile_false(),
            )
            self._capture_impl()
        except BaseException:
            failed = True
            raise
        finally:
            try:
                torch.npu.config.allow_internal_format = previous_format
                if previous_jit:
                    torch.npu.set_compile_mode(jit_compile=True)
            except Exception as exc:
                # set_compile_mode synchronizes; an invalid capture stream can
                # fail again here. Preserve the original capture error on exit.
                if not failed:
                    raise
                logger.warning("[AUDIO_FRONTEND_GRAPH] settings restore failed after capture error: %s", exc)

    def _capture_impl(self) -> None:
        # Each graph owns its allocator pool: runtime batch order is arbitrary,
        # and CNN temporaries must not alias Body graphs on another stream.
        self._stream = torch.npu.Stream()
        self._input_ready = torch.npu.Event()
        self._output_ready = torch.npu.Event()
        self._stream.wait_stream(torch.npu.current_stream())
        logger.info("[AUDIO_FRONTEND_GRAPH] enabled: batches=%s T=%d", self.batches, self.time)
        with torch.npu.stream(self._stream):
            for batch in tqdm(self.batches, desc="Capturing audio CNN graphs", unit="graph"):
                try:
                    inputs = torch.zeros(
                        (batch, 1, self.mel_bins, self.time),
                        dtype=torch.float16,
                        device=self.device,
                    )
                    for _ in range(2):
                        self.forward_eager(inputs)
                    self._stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=self._stream):
                        output = self.forward_eager(inputs)
                    self.runners[batch] = (graph, inputs, output)
                    self._stream.synchronize()
                except Exception as exc:
                    raise RuntimeError(
                        f"Audio CNN graph capture failed at batch={batch}, T={self.time}: {exc}"
                    ) from exc
        logger.info("[AUDIO_FRONTEND_GRAPH] captured batches=%s", self.batches)

    @torch.inference_mode()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        with self._lock:
            self._calls += 1
            reason = None
            if features.ndim != 4 or tuple(features.shape[1:]) != (1, self.mel_bins, self.time):
                reason = "shape_mismatch"
            elif features.dtype != torch.float16 or features.device != self.device:
                reason = "dtype_or_device"
            elif not self.runners or features.shape[0] == 0:
                reason = "not_captured_or_empty"
            if reason:
                logger.info(
                    "[AUDIO_FRONTEND_GRAPH] call=%d chunks=%d T=%d graphs=[] padding=0 fallback=%s",
                    self._calls,
                    features.shape[0],
                    features.shape[-1],
                    reason,
                )
                return self.forward_eager(features)

            plan = plan_frontend_batches(features.shape[0], self.batches)
            caller = torch.npu.current_stream()
            self._input_ready.record(caller)
            self._stream.wait_event(self._input_ready)
            # The packed feature tensor is allocated on the caller's stream;
            # keep its storage live until the asynchronous input copy completes.
            features.record_stream(self._stream)
            with torch.npu.stream(self._stream):
                outputs = []
                offset = 0
                for real, batch in plan:
                    graph, inputs, output = self.runners[batch]
                    inputs[:real].copy_(features[offset : offset + real])
                    if real < batch:
                        inputs[real:].zero_()
                    graph.replay()
                    outputs.append(output[:real].clone())
                    offset += real
                result = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
                self._output_ready.record(self._stream)
            caller.wait_event(self._output_ready)
            result.record_stream(caller)
            logger.info(
                "[AUDIO_FRONTEND_GRAPH] call=%d chunks=%d T=%d graphs=%s padding=%d fallback=none state=submitted",
                self._calls,
                features.shape[0],
                self.time,
                [batch for _, batch in plan],
                sum(batch - real for real, batch in plan),
            )
            return result

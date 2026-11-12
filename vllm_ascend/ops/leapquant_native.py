# SPDX-License-Identifier: Apache-2.0
"""Explicit loader for protected AIV LeapQuant on the current NPU stream.

Only the padded LQ04 serving ABI is accepted. Importing this module does not
build or load a library. Enable it with additional_config.leapquant_config.
"""

import ctypes
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - register the NPU backend before using torch.npu

PACKET_BYTES = 27744
SERVING_PACKET_BYTES = 32768
FORMAT_TAG = 0x4C513034


class NativeLeapQuant:
    """W16/R4 recurrent state with INT8 residual and FP16 factors/records."""

    def __init__(self, library: str, cores: int = 40):
        if type(cores) is not int or not 1 <= cores <= 48:
            raise ValueError("cores must be in [1,48]")
        self.cores = cores
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        pointer, integer = ctypes.c_void_p, ctypes.c_uint32
        try:
            metadata = []
            for symbol in ("leapquant_packet_bytes", "leapquant_format_tag"):
                function = getattr(self.library, symbol)
                function.argtypes, function.restype = [], integer
                metadata.append(function())
            if metadata != [PACKET_BYTES, FORMAT_TAG]:
                raise ValueError("LeapQuant library packet format mismatch")
            for attribute, symbol, pointers, integers in (
                ("_layout_encode", "leapquant_layout_encode_launch", 4, 4),
                ("_layout_export", "leapquant_layout_export_launch", 5, 4),
                ("_padded_encode", "leapquant_padded_encode_launch", 4, 4),
                ("_padded_decode", "leapquant_padded_decode_launch", 9, 5),
                ("_padded_export", "leapquant_padded_export_launch", 5, 4),
            ):
                launch = getattr(self.library, symbol)
                launch.argtypes = [integer] + [pointer] * pointers + [integer] * integers
                launch.restype = None
                setattr(self, attribute, launch)
        except AttributeError as error:
            raise ValueError("LeapQuant library ABI is obsolete; rebuild it") from error

    @staticmethod
    def _packet(packet, indices, batch, heads):
        if (
            packet.dtype != torch.uint8
            or packet.ndim != 3
            or packet.shape[1] != heads
            or packet.shape[2] != SERVING_PACKET_BYTES
        ):
            raise ValueError("native LeapQuant requires uint8 [blocks, heads, 32768] packets")
        if indices.dtype != torch.int32 or indices.shape != (batch,):
            raise ValueError("native LeapQuant requires int32 [requests] indices")
        if not packet.is_contiguous() or not indices.is_contiguous():
            raise ValueError("native LeapQuant packet/indices must be contiguous")
        if packet.device != indices.device or packet.device.type != "npu":
            raise ValueError("native LeapQuant inputs must reside on the same NPU")
        if heads != 32:
            raise ValueError("native LeapQuant requires 32 value heads (TP1)")
        if packet.shape[0] < 1 or batch < 1 or heads < 1:
            raise ValueError("native LeapQuant requires nonempty cache and batch")

    @staticmethod
    def _stream(device):
        if torch.npu.current_device() != device.index:
            raise ValueError("the current NPU must match the input device")
        return torch.npu.current_stream(device).npu_stream

    def _layout_launch(self, name, pointers, batch, heads, packet):
        launch = getattr(self, name)
        launch(self.cores, self._stream(packet.device), *pointers, batch, heads, packet.shape[0], packet.shape[-1])

    def encode_state(self, state, packet, indices, prefill_layout=False):
        """Encode FP32 states without a full FP32 HBM scratch allocation."""
        if state.ndim != 4 or state.shape[-2:] != (128, 128) or state.dtype != torch.float32:
            raise ValueError("native LeapQuant state must be FP32 [requests, heads, 128, 128]")
        if not state.is_contiguous() or state.device != packet.device:
            raise ValueError("native LeapQuant state must be contiguous on the packet NPU")
        batch, heads = state.shape[:2]
        self._packet(packet, indices, batch, heads)
        launch = self._padded_encode
        if prefill_layout:
            self._layout_launch(
                "_layout_encode",
                [state.data_ptr(), packet.data_ptr(), indices.data_ptr()],
                batch,
                heads,
                packet,
            )
            return
        launch(
            self.cores,
            self._stream(packet.device),
            state.data_ptr(),
            packet.data_ptr(),
            indices.data_ptr(),
            batch,
            heads,
            packet.shape[0],
            packet.shape[-1],
        )

    def export_state(self, packet, indices, has_state, prefill_layout=False):
        """Materialize only selected prefill states, including pending records.

        False mask entries yield zero without reading the potentially reused
        slot. Output uses native [V,K] orientation; the prefill op uses [K,V].
        """
        batch = indices.numel()
        self._packet(packet, indices, batch, 32)
        if (
            has_state.shape != (batch,)
            or has_state.dtype != torch.bool
            or has_state.device != packet.device
            or not has_state.is_contiguous()
        ):
            raise ValueError("export requires contiguous bool [requests] mask on the packet NPU")
        output = torch.empty((batch, 32, 128, 128), dtype=torch.float32, device=packet.device)
        launch = self._padded_export
        if prefill_layout:
            self._layout_launch(
                "_layout_export",
                [packet.data_ptr(), indices.data_ptr(), has_state.data_ptr(), output.data_ptr()],
                batch,
                32,
                packet,
            )
            return output
        launch(
            self.cores,
            self._stream(packet.device),
            packet.data_ptr(),
            indices.data_ptr(),
            has_state.data_ptr(),
            output.data_ptr(),
            batch,
            32,
            packet.shape[0],
            packet.shape[-1],
        )
        return output

    def _validate_decode(self, q, k, v, beta, g, packet, indices):
        if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
            raise ValueError("native LeapQuant Q/K/V must have [requests, heads, dim] shape")
        batch, heads = v.shape[:2]
        if q.shape[1] != 16:
            raise ValueError("native LeapQuant requires 16 key heads (TP1)")
        if q.shape[0] != batch or q.shape[-1] != 128 or v.shape[-1] != 128 or q.shape[1] == 0 or heads % q.shape[1]:
            raise ValueError("native LeapQuant requires head dimension 128 and integral GQA")
        self._packet(packet, indices, batch, heads)
        if beta.shape != (batch, heads) or g.shape != (batch, heads):
            raise ValueError("native LeapQuant requires scalar GDN gates")
        if any(t.dtype != torch.bfloat16 for t in (q, k, v, beta)) or g.dtype != torch.float32:
            raise ValueError("native LeapQuant requires BF16 Q/K/V/beta and FP32 log forget gates")
        if any(t.device != packet.device or not t.is_contiguous() for t in (q, k, v, beta, g)):
            raise ValueError("native LeapQuant inputs must be contiguous on the packet NPU")
        return batch, heads

    def decode(self, q, k, v, beta, g, packet, indices):
        """Update one BF16 token per request, including any required fold."""
        batch, heads = self._validate_decode(q, k, v, beta, g, packet, indices)
        output = torch.empty_like(v)
        launch = self._padded_decode
        launch(
            self.cores,
            self._stream(packet.device),
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            beta.data_ptr(),
            g.data_ptr(),
            packet.data_ptr(),
            indices.data_ptr(),
            output.data_ptr(),
            batch,
            heads,
            q.shape[1],
            packet.shape[0],
            packet.shape[-1],
        )
        return output

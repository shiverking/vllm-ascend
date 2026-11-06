# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in loader for the isolated AscendC LeapQuant experiment.

The native v4 packet is distinct from the Triton v1 representation. No library
is built or loaded by importing this module. It is not dispatched by models.
"""

import ctypes
from pathlib import Path

import torch

PACKET_BYTES = 27744
FORMAT_TAG = 0x4C513034


class NativeLeapQuant:
    """Load a separately built kernel library on the current NPU stream."""

    def __init__(self, library: str, cores: int = 40):
        if not 1 <= cores <= 48:
            raise ValueError("cores must be in [1, 48] for the 910B exploration")
        self.cores = cores
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        pointer, integer = ctypes.c_void_p, ctypes.c_uint32
        metadata = []
        for symbol in ("leapquant_packet_bytes", "leapquant_format_tag", "leapquant_power_iterations"):
            try:
                function = getattr(self.library, symbol)
            except AttributeError as error:
                raise ValueError("native LeapQuant library ABI is obsolete; rebuild it") from error
            function.argtypes, function.restype = [], integer
            metadata.append(function())
        if metadata[:2] != [PACKET_BYTES, FORMAT_TAG]:
            raise ValueError("native LeapQuant library packet format does not match this loader")
        self.power_iterations = metadata[2]
        self._encode = self.library.leapquant_encode_launch
        self._encode.argtypes = [integer] + [pointer] * 4 + [integer] * 3
        self._encode.restype = None
        self._decode = self.library.leapquant_decode_launch
        self._decode.argtypes = [integer] + [pointer] * 9 + [integer] * 4
        self._decode.restype = None

    @staticmethod
    def _packet(packet, indices, batch, heads):
        if packet.dtype != torch.uint8 or packet.ndim != 3 or packet.shape[1:] != (heads, PACKET_BYTES):
            raise ValueError("native LeapQuant requires uint8 [blocks, heads, 27744] packets")
        if indices.dtype != torch.int32 or indices.shape != (batch,):
            raise ValueError("native LeapQuant requires int32 [requests] indices")
        if not packet.is_contiguous() or not indices.is_contiguous():
            raise ValueError("native LeapQuant packet/indices must be contiguous")
        if packet.device != indices.device or packet.device.type != "npu":
            raise ValueError("native LeapQuant inputs must reside on the same NPU")
        if heads != 32:
            raise ValueError("native LeapQuant prototype requires 32 value heads (TP1)")
        if packet.shape[0] < 1 or batch < 1 or heads < 1:
            raise ValueError("native LeapQuant requires nonempty cache and batch")

    @staticmethod
    def _stream(device):
        if torch.npu.current_device() != device.index:
            raise ValueError("the current NPU must match the input device")
        return torch.npu.current_stream(device).npu_stream

    def encode_state(self, state, packet, indices):
        """Encode FP32 states without a full FP32 HBM scratch allocation."""
        if state.ndim != 4 or state.shape[-2:] != (128, 128) or state.dtype != torch.float32:
            raise ValueError("native LeapQuant state must be FP32 [requests, heads, 128, 128]")
        if not state.is_contiguous() or state.device != packet.device:
            raise ValueError("native LeapQuant state must be contiguous on the packet NPU")
        batch, heads = state.shape[:2]
        self._packet(packet, indices, batch, heads)
        self._encode(
            self.cores,
            self._stream(packet.device),
            state.data_ptr(),
            packet.data_ptr(),
            indices.data_ptr(),
            batch,
            heads,
            packet.shape[0],
        )

    def decode(self, q, k, v, beta, g, packet, indices):
        """Update one BF16 token per request, including any required fold."""
        if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
            raise ValueError("native LeapQuant Q/K/V must have [requests, heads, dim] shape")
        batch, heads = v.shape[:2]
        if q.shape[1] != 16:
            raise ValueError("native LeapQuant prototype requires 16 key heads (TP1)")
        if q.shape[0] != batch or q.shape[-1] != 128 or v.shape[-1] != 128 or q.shape[1] == 0 or heads % q.shape[1]:
            raise ValueError("native LeapQuant requires head dimension 128 and integral GQA")
        self._packet(packet, indices, batch, heads)
        if beta.shape != (batch, heads) or g.shape != (batch, heads):
            raise ValueError("native LeapQuant requires scalar GDN gates")
        if any(t.dtype != torch.bfloat16 for t in (q, k, v, beta)) or g.dtype != torch.float32:
            raise ValueError("native LeapQuant requires BF16 Q/K/V/beta and FP32 log forget gates")
        if any(t.device != packet.device or not t.is_contiguous() for t in (q, k, v, beta, g)):
            raise ValueError("native LeapQuant inputs must be contiguous on the packet NPU")
        output = torch.empty_like(v)
        self._decode(
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
        )
        return output

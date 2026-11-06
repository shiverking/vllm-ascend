# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in loader for the isolated AscendC LeapQuant experiment.

The native v4 packet is distinct from the Triton v1 representation. No library
is built or loaded by importing this module. It is not dispatched by models.
"""

import ctypes
from pathlib import Path

import torch
import torch_npu

PACKET_BYTES = 27744
FORMAT_TAG = 0x4C513034


class NativeLeapQuant:
    """Load a separately built kernel library on the current NPU stream."""

    def __init__(self, library: str, cores: int = 40, value_tile: int = 128, streaming: bool = False):
        if not 1 <= cores <= 48:
            raise ValueError("cores must be in [1, 48] for the 910B exploration")
        if value_tile not in (32, 64, 128):
            raise ValueError("value_tile must be 32, 64 or 128")
        if streaming and value_tile != 128:
            raise ValueError("streaming requires value_tile=128")
        self.streaming = streaming
        self.value_tile = value_tile
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
        self._decode = (
            self.library.leapquant_decode_launch if value_tile == 128 else self.library.leapquant_tiled_launch
        )
        if streaming:
            self._decode = self.library.leapquant_stream_launch
        self._decode.argtypes = [integer] + [pointer] * 9 + [integer] * (4 if value_tile == 128 else 5)
        self._decode.restype = None
        for attribute, symbol in (
            ("_pack_digits", "leapquant_digit_pack_launch"),
            ("_unpack_digits", "leapquant_digit_unpack_launch"),
        ):
            launch = getattr(self.library, symbol, None)
            if launch is not None:
                launch.argtypes = [integer] + [pointer] * 4 + [integer]
                launch.restype = None
            setattr(self, attribute, launch)
        self._transpose_residual = getattr(self.library, "leapquant_residual_transpose_launch", None)
        if self._transpose_residual is not None:
            self._transpose_residual.argtypes = [integer] + [pointer] * 3 + [integer] * 2
            self._transpose_residual.restype = None

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
            *(() if self.value_tile == 128 else (self.value_tile,)),
        )
        return output

    def pack_query_digits(self, query):
        """Prepare four signed radix-128 digits for each of two FP32 vectors."""
        if query.ndim != 3 or query.shape[1:] != (2, 128) or query.shape[0] < 1:
            raise ValueError("digit queries require [heads, 2, 128]")
        if query.dtype != torch.float32 or query.device.type != "npu" or not query.is_contiguous():
            raise ValueError("digit queries must be contiguous FP32 NPU tensors")
        if torch_npu.get_npu_format(query) != 2:
            raise ValueError("digit queries require ND layout")
        if self._pack_digits is None:
            raise ValueError("rebuild the native library for digit helpers")
        packed = torch.empty((query.shape[0], 16, 128), dtype=torch.int8, device=query.device)
        scales = torch.empty((query.shape[0], 8), dtype=torch.float32, device=query.device)
        launch = self._pack_digits
        launch(
            self.cores,
            self._stream(query.device),
            query.data_ptr(),
            packed.data_ptr(),
            scales.data_ptr(),
            query.shape[0],
        )
        return packed, scales

    def unpack_query_products(self, products, scales):
        """Reconstruct FP32 residual contractions from exact Cube INT32 sums."""
        if products.ndim != 3 or products.shape[1:] != (16, 128) or products.shape[0] < 1:
            raise ValueError("digit products require [heads, 16, 128]")
        if products.dtype != torch.int32 or scales.dtype != torch.float32 or scales.shape != (products.shape[0], 8):
            raise ValueError("digit products/scales require INT32 and FP32 types with matching heads")
        if (
            products.device.type != "npu"
            or scales.device != products.device
            or not products.is_contiguous()
            or not scales.is_contiguous()
        ):
            raise ValueError("digit products/scales must be contiguous on the same NPU")
        if torch_npu.get_npu_format(products) != 2 or torch_npu.get_npu_format(scales) != 2:
            raise ValueError("digit products/scales require ND layout")
        if self._unpack_digits is None:
            raise ValueError("rebuild the native library for digit helpers")
        output = torch.empty((products.shape[0], 2, 128), dtype=torch.float32, device=products.device)
        launch = self._unpack_digits
        launch(
            self.cores,
            self._stream(products.device),
            products.data_ptr(),
            scales.data_ptr(),
            output.data_ptr(),
            products.shape[0],
        )
        return output

    def transpose_residual(self, state):
        """Lossless INT8 transpose; supports ND residual views inside AoS packets."""
        if state.ndim != 3 or state.shape[1:] != (128, 128) or state.shape[0] < 1:
            raise ValueError("residual transpose requires [heads, 128, 128]")
        if state.dtype != torch.int8 or state.device.type != "npu":
            raise ValueError("residual transpose requires an INT8 NPU tensor")
        if state.stride(1) != 128 or state.stride(2) != 1 or not 0 <= state.stride(0) < 2**32:
            raise ValueError("residual transpose requires contiguous matrix rows")
        if torch_npu.get_npu_format(state) != 2:
            raise ValueError("residual transpose requires ND layout")
        if self._transpose_residual is None:
            raise ValueError("rebuild the native library for residual transpose")
        output = torch.empty(state.shape, dtype=state.dtype, device=state.device)
        self._transpose_residual(
            self.cores, self._stream(state.device), state.data_ptr(), output.data_ptr(), state.shape[0], state.stride(0)
        )
        return output

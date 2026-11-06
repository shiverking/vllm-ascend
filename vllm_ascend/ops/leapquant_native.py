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
SERVING_PACKET_BYTES = 32768
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
        self._gdn_pack = getattr(self.library, "leapquant_gdn_pack_launch", None)
        if self._gdn_pack is not None:
            self._gdn_pack.argtypes = [integer] + [pointer] * 7 + [integer] * 3
            self._gdn_pack.restype = None
        self._cube_post = getattr(self.library, "leapquant_cube_post_launch", None)
        if self._cube_post is not None:
            self._cube_post.argtypes = [integer] + [pointer] * 12 + [integer] * 5
            self._cube_post.restype = None

        self._export = getattr(self.library, "leapquant_export_launch", None)
        if self._export is not None:
            self._export.argtypes = [integer] + [pointer] * 5 + [integer] * 3
            self._export.restype = None

        for attribute, symbol, pointers, integers in (
            ("_padded_encode", "leapquant_padded_encode_launch", 4, 4),
            ("_padded_decode", "leapquant_padded_decode_launch", 9, 5),
            ("_padded_export", "leapquant_padded_export_launch", 5, 4),
        ):
            launch = getattr(self.library, symbol, None)
            if launch is not None:
                launch.argtypes = [integer] + [pointer] * pointers + [integer] * integers
                launch.restype = None
            setattr(self, attribute, launch)

    @staticmethod
    def _packet(packet, indices, batch, heads):
        if (
            packet.dtype != torch.uint8
            or packet.ndim != 3
            or packet.shape[1] != heads
            or packet.shape[2] not in (PACKET_BYTES, SERVING_PACKET_BYTES)
        ):
            raise ValueError("native LeapQuant requires uint8 [blocks, heads, 27744 or 32768] packets")
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
        padded = packet.shape[-1] == SERVING_PACKET_BYTES
        launch = self._padded_encode if padded else self._encode
        if launch is None:
            raise ValueError("rebuild the native library for padded serving packets")
        launch(
            self.cores,
            self._stream(packet.device),
            state.data_ptr(),
            packet.data_ptr(),
            indices.data_ptr(),
            batch,
            heads,
            packet.shape[0],
            *((packet.shape[-1],) if padded else ()),
        )

    def export_state(self, packet, indices, has_state):
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
        if self._export is None:
            raise ValueError("rebuild the native library for prefill state export")
        output = torch.empty((batch, 32, 128, 128), dtype=torch.float32, device=packet.device)
        padded = packet.shape[-1] == SERVING_PACKET_BYTES
        launch = self._padded_export if padded else self._export
        if launch is None:
            raise ValueError("rebuild the native library for padded serving packets")
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
            *((packet.shape[-1],) if padded else ()),
        )
        return output

    def _validate_decode(self, q, k, v, beta, g, packet, indices):
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
        return batch, heads

    def decode(self, q, k, v, beta, g, packet, indices):
        """Update one BF16 token per request, including any required fold."""
        batch, heads = self._validate_decode(q, k, v, beta, g, packet, indices)
        output = torch.empty_like(v)
        padded = packet.shape[-1] == SERVING_PACKET_BYTES
        if padded and (self.value_tile != 128 or self.streaming or self._padded_decode is None):
            raise ValueError("padded serving packets require the fused serving library")
        launch = self._padded_decode if padded else self._decode
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
            *((packet.shape[-1],) if padded else (() if self.value_tile == 128 else (self.value_tile,))),
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

    def transpose_residual(self, state, out=None):
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
        if out is not None and (
            out.shape != state.shape
            or out.dtype != torch.int8
            or out.device != state.device
            or not out.is_contiguous()
            or torch_npu.get_npu_format(out) != 2
        ):
            raise ValueError("residual transpose output must be matching contiguous INT8 ND")
        output = out if out is not None else torch.empty(state.shape, dtype=state.dtype, device=state.device)
        self._transpose_residual(
            self.cores, self._stream(state.device), state.data_ptr(), output.data_ptr(), state.shape[0], state.stride(0)
        )
        return output

    def prepare_gdn_queries(self, q, k, packet, indices):
        """Read model BF16 Q/K and checkpoint smoothing directly into digits."""
        if q.ndim != 3 or q.shape[1:] != (16, 128) or k.shape != q.shape:
            raise ValueError("GDN preparation requires [batch,16,128] Q/K")
        self._packet(packet, indices, q.shape[0], 32)
        if packet.shape[-1] != PACKET_BYTES:
            raise ValueError("Cube GDN requires unpadded isolated packets")
        if any(t.dtype != torch.bfloat16 or t.device != packet.device or not t.is_contiguous() for t in (q, k)):
            raise ValueError("GDN preparation requires matching contiguous BF16 Q/K")
        if self._gdn_pack is None:
            raise ValueError("rebuild the native library for Cube GDN")
        packed = torch.empty((q.shape[0] * 32, 16, 128), dtype=torch.int8, device=q.device)
        scales = torch.empty((q.shape[0] * 32, 8), dtype=torch.float32, device=q.device)
        self._gdn_pack(
            self.cores,
            self._stream(q.device),
            q.data_ptr(),
            k.data_ptr(),
            packet.data_ptr(),
            indices.data_ptr(),
            packed.data_ptr(),
            scales.data_ptr(),
            q.shape[0],
            16,
            packet.shape[0],
        )
        return packed, scales

    def decode_cube_products(self, q, k, v, beta, g, packet, indices, products, scales, weights, refresh_layout=True):
        """Finish the complete GDN recurrence, including any fold and cache refresh."""
        batch, heads = self._validate_decode(q, k, v, beta, g, packet, indices)
        if packet.shape[-1] != PACKET_BYTES:
            raise ValueError("Cube GDN requires unpadded isolated packets")
        if products.shape != (batch * heads, 16, 128) or products.dtype != torch.int32:
            raise ValueError("Cube products must be INT32 [batch*heads,16,128]")
        if scales.shape != (batch * heads, 8) or scales.dtype != torch.float32:
            raise ValueError("Cube query scales must be FP32 [batch*heads,8]")
        if weights.shape != (packet.shape[0] * heads, 128, 128) or weights.dtype != torch.int8:
            raise ValueError("Cube weights must match the packet pool")
        if any(
            t.device != packet.device or not t.is_contiguous() or torch_npu.get_npu_format(t) != 2
            for t in (products, scales, weights)
        ):
            raise ValueError("Cube intermediate tensors must be contiguous ND on the packet NPU")
        if self._cube_post is None:
            raise ValueError("rebuild the native library for Cube GDN")
        output = torch.empty_like(v)
        self._cube_post(
            self.cores,
            self._stream(q.device),
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            beta.data_ptr(),
            g.data_ptr(),
            packet.data_ptr(),
            indices.data_ptr(),
            output.data_ptr(),
            products.data_ptr(),
            scales.data_ptr(),
            weights.data_ptr(),
            batch,
            heads,
            16,
            packet.shape[0],
            int(refresh_layout),
        )
        return output


class NativeCubeGDN:
    """Isolated TP1 GDN recurrence with owned, identity-mapped cache slots.

    Q/K/V, beta and log gates are layer inputs. This does not include the
    model's projections, convolution, normalization or output projection.
    Both packet and weight buffers form the checkpoint and must be restored.
    """

    def __init__(self, backend: NativeLeapQuant, state):
        if backend.value_tile != 128 or backend.streaming:
            raise ValueError("Cube GDN requires the fused native backend")
        if (
            state.ndim != 4
            or state.shape[1:] != (32, 128, 128)
            or state.dtype != torch.float32
            or state.device.type != "npu"
        ):
            raise ValueError("Cube GDN state must be FP32 NPU [batch,32,128,128]")
        self.backend = backend
        self.packet = torch.empty((state.shape[0], 32, PACKET_BYTES), dtype=torch.uint8, device=state.device)
        self.indices = torch.arange(state.shape[0], dtype=torch.int32, device=state.device)
        self.weights = torch.empty((state.shape[0] * 32, 128, 128), dtype=torch.int8, device=state.device)
        self.unit = torch.ones(128, dtype=torch.float32, device=state.device)
        backend.encode_state(state, self.packet, self.indices)
        self.refresh_layout()

    def residual_view(self):
        return self.packet.view(torch.int8)[..., :16384].reshape(-1, 128, 128)

    def refresh_layout(self):
        """Update the fixed weight buffer; used only for initialization or diagnostics."""
        self.backend.transpose_residual(self.residual_view(), out=self.weights)

    def step(self, q, k, v, beta, g):
        return self._step(q, k, v, beta, g, True)

    def _step(self, q, k, v, beta, g, refresh_layout):
        # The false case is a counterfactual benchmark only: the caller must
        # refresh the layout after each complete 16-token graph window.
        if q.shape[0] != self.packet.shape[0]:
            raise ValueError("Cube GDN batch must match its owned cache")
        self.backend._validate_decode(q, k, v, beta, g, self.packet, self.indices)
        packed, scales = self.backend.prepare_gdn_queries(q, k, self.packet, self.indices)
        products = torch_npu.npu_quant_matmul(packed, self.weights, self.unit, output_dtype=torch.int32)
        return self.backend.decode_cube_products(
            q, k, v, beta, g, self.packet, self.indices, products, scales, self.weights, refresh_layout
        )

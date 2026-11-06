# SPDX-License-Identifier: Apache-2.0
"""Experimental GDN state quantization for Ascend (attention KV stays BF16).

Each temporal cache row is a self-contained byte packet. This preserves the
existing temporal-state copy contract, including partially filled windows.
Only 128x128 scalar-gated GDN, standard single-token decode is supported.
"""

import torch
from vllm.triton_utils import tl, triton

DIM: tl.constexpr = 128
WINDOW: tl.constexpr = 16
RANK: tl.constexpr = 4
BASE: tl.constexpr = 0
SCALE: tl.constexpr = 16384
SMOOTH: tl.constexpr = 16896
LEFT: tl.constexpr = 17408
RIGHT: tl.constexpr = 18432
KEYS: tl.constexpr = 19456
VALUES: tl.constexpr = 23552
DECAY: tl.constexpr = 27648
CURSOR: tl.constexpr = 27652
PACKET_BYTES: tl.constexpr = 27680


V_TILE: tl.constexpr = 8
FIT_ITERS: tl.constexpr = 6
QK_SCALE: tl.constexpr = DIM**-0.5


@triton.jit
def _encode(R, p8, p16, p32, pi32):
    col, rr = tl.arange(0, 128), tl.arange(0, 8)
    # Fixed byte offsets below implement the packet layout declared above.
    # Keep only an 8x128 tile in UB. R is private fold scratch, never a cache.
    for r in range(4):
        direction = tl.full((128,), 1.0, tl.float32)
        for _iteration in range(6):
            accum = tl.full((128,), 0.0, tl.float32)
            for tile in range(16):
                row = tile * 8 + rr
                matrix = tl.load(R + row[:, None] * 128 + col[None, :])
                u = tl.sum(matrix * direction[None, :], 1)
                accum += tl.sum(matrix * u[:, None], 0)
            direction = accum * tl.rsqrt(tl.maximum(tl.sum(accum * accum, 0), 1.0e-24))
        vh = direction.to(tl.float16).to(tl.float32)
        tl.store(p16 + 9216 + r * 128 + col, vh)
        for tile in range(16):
            row = tile * 8 + rr
            matrix = tl.load(R + row[:, None] * 128 + col[None, :])
            uh = tl.sum(matrix * direction[None, :], 1).to(tl.float16).to(tl.float32)
            tl.store(p16 + 8704 + row * 4 + r, uh)
            tl.store(R + row[:, None] * 128 + col[None, :], matrix - uh[:, None] * vh[None, :])
    smooth = tl.full((128,), 1.0e-12, tl.float32)
    for tile in range(16):
        row = tile * 8 + rr
        matrix = tl.load(R + row[:, None] * 128 + col[None, :])
        smooth = tl.maximum(smooth, tl.max(tl.abs(matrix), 0))
    tl.store(p32 + 4224 + col, smooth)
    for tile in range(16):
        row = tile * 8 + rr
        matrix = tl.load(R + row[:, None] * 128 + col[None, :]) / smooth[None, :]
        scale = tl.maximum(tl.max(tl.abs(matrix), 1), 1.0e-12) / 127.0
        quant = matrix / scale[:, None]
        quant = tl.where(quant >= 0, tl.floor(quant + 0.5), -tl.floor(-quant + 0.5))
        tl.store(p8 + row[:, None] * 128 + col[None, :], tl.minimum(tl.maximum(quant, -127.0), 127.0).to(tl.int8))
        tl.store(p32 + 4096 + row, scale)
    tl.store(p32 + 6912, 1.0)
    tl.store(pi32 + 6913, 0)


@triton.jit
def _encode_kernel(Scratch, P8, P16, P32, PI32, Indices, H: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    index = tl.load(Indices + b).to(tl.int64)
    offset = (index * H + h) * 27680
    _encode(Scratch + (b * H + h) * 16384, P8 + offset, P16 + offset // 2, P32 + offset // 4, PI32 + offset // 4)
    # Inactive record slots are masked by the zero cursor; clearing is not
    # needed for correctness and would add global writes at each fold.


@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    Beta,
    G,
    P8,
    P16,
    P32,
    PI32,
    Indices,
    Out,
    Scratch,
    H: tl.constexpr,
    HK: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BS: tl.constexpr,
    GS: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    index = tl.load(Indices + b).to(tl.int64)
    rr, col = tl.arange(0, 8), tl.arange(0, 128)
    if index >= 0:
        offset = (index * H + h) * 27680
        p8, p16, p32, pi32 = P8 + offset, P16 + offset // 2, P32 + offset // 4, PI32 + offset // 4
        rank, win = tl.arange(0, 4), tl.arange(0, 16)
        count = tl.load(pi32 + 6913)
        alpha = tl.exp(tl.load(G + b * GS + h).to(tl.float32))
        decay = tl.load(p32 + 6912) * alpha
        q = tl.load(Q + b * QS + (h // (H // HK)) * 128 + col).to(tl.float32) * 0.08838834764831845
        k = tl.load(K + b * KS + (h // (H // HK)) * 128 + col).to(tl.float32)
        beta = tl.load(Beta + b * BS + h).to(tl.float32)
        smooth = tl.load(p32 + 4224 + col)
        right = tl.load(p16 + 9216 + rank[:, None] * 128 + col[None, :]).to(tl.float32)
        keys = tl.load(p16 + 9728 + win[:, None] * 128 + col[None, :], win[:, None] < count, 0).to(tl.float32)
        wk = tl.sum(keys * k[None, :], 1)
        rk, rq = tl.sum(right * k[None, :], 1), tl.sum(right * q[None, :], 1)
        keys = tl.where(win[:, None] == count, k[None, :], keys)
        wq = tl.sum(keys * q[None, :], 1)
        tl.store(p16 + 9728 + count * 128 + col, k)
        for tile in range(16):
            row = tile * 8 + rr
            value = tl.load(V + b * VS + h * 128 + row).to(tl.float32)
            residual = tl.load(p8 + row[:, None] * 128 + col[None, :]).to(tl.float32)
            scale = tl.load(p32 + 4096 + row)
            left = tl.load(p16 + 8704 + row[:, None] * 4 + rank[None, :]).to(tl.float32)
            values = tl.load(p16 + 11776 + row[:, None] * 16 + win[None, :], win[None, :] < count, 0).to(tl.float32)
            values = (values * alpha).to(tl.float16).to(tl.float32)
            bk = tl.sum(residual * (smooth * k)[None, :], 1) * scale + tl.sum(left * rk[None, :], 1)
            predicted = bk * decay + tl.sum(values * wk[None, :], 1)
            delta = ((value - predicted) * beta).to(tl.float16).to(tl.float32)
            values = tl.where(win[None, :] == count, delta[:, None], values)
            bq = tl.sum(residual * (smooth * q)[None, :], 1) * scale + tl.sum(left * rq[None, :], 1)
            output = bq * decay + tl.sum(values * wq[None, :], 1)
            tl.store(Out + (b * H + h) * 128 + row, output)
            if count == 15:
                matrix = residual * scale[:, None] * smooth[None, :]
                for r in range(4):
                    lv = tl.sum(tl.where(rank[None, :] == r, left, 0.0), 1)
                    rv = tl.sum(tl.where(rank[:, None] == r, right, 0.0), 0)
                    matrix += lv[:, None] * rv[None, :]
                matrix *= decay
                for w in range(16):
                    lv = tl.sum(tl.where(win[None, :] == w, values, 0.0), 1)
                    rv = tl.sum(tl.where(win[:, None] == w, keys, 0.0), 0)
                    matrix += lv[:, None] * rv[None, :]
                tl.store(Scratch + (b * H + h) * 16384 + row[:, None] * 128 + col[None, :], matrix)
            else:
                tl.store(p16 + 11776 + row[:, None] * 16 + win[None, :], values)
        if count == 15:
            _encode(Scratch + (b * H + h) * 16384, p8, p16, p32, pi32)
        else:
            tl.store(p32 + 6912, decay)
            tl.store(pi32 + 6913, count + 1)
    else:
        for tile in range(16):
            tl.store(Out + (b * H + h) * 128 + tile * 8 + rr, 0.0)


def _validate_packet(packet, indices, batch, heads):
    if packet.dtype != torch.uint8 or packet.ndim != 3 or packet.shape[1:] != (heads, PACKET_BYTES):
        raise ValueError("LeapQuant requires a uint8 [blocks, heads, 27680] packet")
    if not packet.is_contiguous():
        raise ValueError("LeapQuant packet must be contiguous")
    if indices.dtype != torch.int32 or indices.shape != (batch,) or not indices.is_contiguous():
        raise ValueError("LeapQuant requires contiguous int32 [requests] state indices")
    if packet.device != indices.device:
        raise ValueError("LeapQuant packet and indices must share a device")


def encode_state(state, packet, indices):
    """Encode contiguous FP32 states; scratch is used only at window folds."""
    if state.ndim != 4 or state.shape[-2:] != (DIM, DIM) or state.dtype != torch.float32:
        raise ValueError("LeapQuant encoding requires FP32 [requests, heads, 128, 128] state")
    if not state.is_contiguous() or state.device != packet.device:
        raise ValueError("LeapQuant encoding requires contiguous state on the packet device")
    _validate_packet(packet, indices, state.shape[0], state.shape[1])
    # Encoding only accepts valid slots; negative padding is supported by decode.
    scratch = state.clone()
    _encode_kernel[(state.shape[0], state.shape[1])](
        scratch,
        packet.view(torch.int8),
        packet.view(torch.float16),
        packet.view(torch.float32),
        packet.view(torch.int32),
        indices,
        state.shape[1],
        disable_tightly_coupled_buffer_reuse=True,
    )


def decode(q, k, v, beta, g, packet, indices, scratch=None):
    """One normalized token per request. Provide persistent scratch for graphs."""
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError("LeapQuant expects [requests, heads, dim] Q/K/V tensors")
    batch, heads = v.shape[:2]
    if q.shape[0] != batch or q.shape[-1] != DIM or v.shape[-1] != DIM or q.shape[1] == 0 or heads % q.shape[1]:
        raise ValueError("LeapQuant requires head dimension 128 and integral GQA grouping")
    if beta.shape != (batch, heads) or g.shape != (batch, heads):
        raise ValueError("LeapQuant requires scalar GDN gates per request and value head")
    for tensor in (q, k, v, beta, g):
        if tensor.device != packet.device or not tensor.is_contiguous():
            raise ValueError("LeapQuant inputs must be contiguous and on the packet device")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("LeapQuant Q/K/V must use BF16")
    if beta.dtype not in (torch.bfloat16, torch.float32) or g.dtype != torch.float32:
        raise ValueError("LeapQuant requires BF16/FP32 beta and FP32 log forget gates")
    _validate_packet(packet, indices, batch, heads)
    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    if scratch is None:
        scratch = torch.empty((*v.shape[:2], 128, 128), device=v.device, dtype=torch.float32)
    if scratch.shape != (batch, heads, DIM, DIM) or scratch.dtype != torch.float32:
        raise ValueError("LeapQuant scratch must be FP32 [requests, heads, 128, 128]")
    if not scratch.is_contiguous() or scratch.device != packet.device:
        raise ValueError("LeapQuant scratch must be contiguous and on the packet device")
    _decode_kernel[(q.shape[0], v.shape[1])](
        q,
        k,
        v,
        beta,
        g,
        packet.view(torch.int8),
        packet.view(torch.float16),
        packet.view(torch.float32),
        packet.view(torch.int32),
        indices,
        output,
        scratch,
        v.shape[1],
        q.shape[1],
        q.stride(0),
        k.stride(0),
        v.stride(0),
        beta.stride(0),
        g.stride(0),
        disable_tightly_coupled_buffer_reuse=True,
    )
    return output

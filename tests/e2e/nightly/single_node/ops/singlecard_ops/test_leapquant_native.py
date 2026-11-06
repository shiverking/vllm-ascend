# SPDX-License-Identifier: Apache-2.0
"""Isolated native library tests. Build in pytest's temporary directory only."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.leapquant_native import PACKET_BYTES, NativeCubeGDN, NativeLeapQuant


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    source = Path(__file__).resolve().parents[6] / "csrc/experimental/leapquant"
    build = tmp_path_factory.mktemp("leapquant_native")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
            "-DLEAPQUANT_POWER_ITERS=8",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cmake", "--build", str(build), "-j4"], check=True, capture_output=True)
    return build / "lib/libleapquant_native.so"


@pytest.fixture(scope="module", params=[(128, False), (64, False), (32, False), (128, True)])
def native(native_library, request):
    torch.npu.set_device(0)
    backend = NativeLeapQuant(str(native_library), value_tile=request.param[0], streaming=request.param[1])
    assert backend.power_iterations == 8
    return backend


def _inputs(batch=2):
    q = torch.nn.functional.normalize(torch.randn(batch, 16, 128), dim=-1).bfloat16().npu()
    k = torch.nn.functional.normalize(torch.randn(batch, 16, 128), dim=-1).bfloat16().npu()
    v = torch.randn(batch, 32, 128).bfloat16().npu()
    beta = torch.rand(batch, 32).bfloat16().npu()
    g = (-torch.rand(batch, 32) * 0.2).npu()
    return q, k, v, beta, g


def test_native_against_fp32_trajectory(native):
    torch.manual_seed(71)
    state = torch.randn(2, 32, 128, 128) * 0.02
    packet = torch.zeros(4, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.tensor([3, 1], dtype=torch.int32, device="npu")
    native.encode_state(state.npu(), packet, indices)
    for _ in range(65):
        q, k, v, beta, g = _inputs()
        qf = q.float().cpu().repeat_interleave(2, 1) * 128**-0.5
        kf = k.float().cpu().repeat_interleave(2, 1)
        state = state * g.cpu().exp()[..., None, None]
        predicted = (state @ kf.unsqueeze(-1)).squeeze(-1)
        delta = (v.float().cpu() - predicted) * beta.float().cpu()[..., None]
        state = state + delta[..., None] * kf.unsqueeze(-2)
        expected = (state @ qf.unsqueeze(-1)).squeeze(-1)
        actual = native.decode(q, k, v, beta, g, packet, indices).float().cpu()
        relative_rms = ((actual - expected).square().mean() / expected.square().mean()).sqrt()
        assert torch.isfinite(actual).all() and relative_rms < 0.02
    torch.testing.assert_close(packet[0], torch.zeros_like(packet[0]), rtol=0, atol=0)


def test_native_checkpoint_and_graph(native):
    torch.manual_seed(72)
    packet = torch.zeros(4, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    native.encode_state(torch.zeros(2, 32, 128, 128, device="npu"), packet, indices)
    inputs = _inputs()
    for _ in range(7):
        native.decode(*inputs, packet, indices)
    # Freeze a partial-window checkpoint; continue two independent copies.
    packet[0].copy_(packet[1])
    checkpoint = packet[0].clone()
    packet[3].copy_(packet[0])
    for tensor in inputs:
        tensor[1].copy_(tensor[0])
    indices[1] = 3
    for _ in range(26):
        out = native.decode(*inputs, packet, indices)
        torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)
    torch.testing.assert_close(packet[0], checkpoint, rtol=0, atol=0)
    saved = packet.clone()
    expected = []
    for _ in range(19):
        expected.append(native.decode(*inputs, packet, indices).clone())
    expected_state = packet.clone()
    packet.copy_(saved)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = native.decode(*inputs, packet, indices)
    packet.copy_(saved)
    for expected_output in expected:
        graph.replay()
        torch.testing.assert_close(out, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(packet, expected_state, rtol=0, atol=0)
    saved = packet.clone()
    indices.fill_(-1)
    graph.replay()
    assert out.abs().max().item() == 0
    torch.testing.assert_close(packet, saved, rtol=0, atol=0)


def test_native_zero_low_rank_and_foreign_packet(native):
    indices = torch.zeros(1, dtype=torch.int32, device="npu")
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    q = torch.full((1, 16, 128), 128**-0.5, dtype=torch.bfloat16, device="npu")
    v = torch.zeros(1, 32, 128, dtype=torch.bfloat16, device="npu")
    beta = torch.zeros(1, 32, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(1, 32, device="npu")
    for value in (0.0, 0.01):
        native.encode_state(torch.full((1, 32, 128, 128), value, device="npu"), packet, indices)
        actual = native.decode(q, q, v, beta, g, packet, indices).float()
        torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=2e-4)
    packet.zero_()  # Missing LQ04 tag: no cache mutation, nonfinite output.
    actual = native.decode(q, q, v, beta, g, packet, indices)
    assert torch.isnan(actual).all()
    torch.testing.assert_close(packet, torch.zeros_like(packet), rtol=0, atol=0)


def test_native_rejects_invalid_metadata(native):
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8)
    indices = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="FP32"):
        native.encode_state(torch.zeros(1, 32, 128, 128, dtype=torch.bfloat16), packet, indices)
    with pytest.raises(ValueError, match="value_tile"):
        NativeLeapQuant("not-loaded", value_tile=16)
    with pytest.raises(ValueError, match="cores"):
        NativeLeapQuant("not-loaded", cores=0)


def test_native_keeps_window_updates_immutable(native):
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.zeros(1, dtype=torch.int32, device="npu")
    native.encode_state(torch.zeros(1, 32, 128, 128, device="npu"), packet, indices)
    inputs = _inputs(1)
    native.decode(*inputs, packet, indices)
    # The first raw FP16 update occupies 128 values at the v3 VALUES offset.
    saved = packet[:, :, 23552:23808].clone()
    inputs[-1].fill_(-0.2)
    native.decode(*inputs, packet, indices)
    torch.testing.assert_close(packet[:, :, 23552:23808], saved, rtol=0, atol=0)


def test_native_rejects_obsolete_library(tmp_path, monkeypatch):
    import vllm_ascend.ops.leapquant_native as module

    path = tmp_path / "obsolete.so"
    path.touch()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _: object())
    with pytest.raises(ValueError, match="obsolete"):
        NativeLeapQuant(str(path))


def test_native_quantizer_matches_paper_smoothing(native):
    torch.manual_seed(409)
    state = torch.randn(1, 32, 128, 128) * torch.linspace(0.01, 0.2, 128)
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.zeros(1, dtype=torch.int32, device="npu")
    native.encode_state(state.npu(), packet, indices)
    stored = packet.cpu()
    left = stored[..., 17408:18432].contiguous().view(torch.float16).float().view(1, 32, 4, 128)
    right = stored[..., 18432:19456].contiguous().view(torch.float16).float().view(1, 32, 4, 128)
    residual = state - left.transpose(-1, -2) @ right
    expected_smooth = residual.abs().mean(-2).sqrt().clamp_min(1e-6)
    smooth = stored[..., 16896:17408].contiguous().view(torch.float32)
    scale = stored[..., 16384:16896].contiguous().view(torch.float32)
    torch.testing.assert_close(smooth, expected_smooth, rtol=2e-5, atol=1e-7)
    normalized = residual / smooth.unsqueeze(-2)
    maximum = normalized.abs().amax(-1)
    expected_scale = torch.where(maximum > 1e-30, maximum / 127, 1.0)
    torch.testing.assert_close(scale, expected_scale, rtol=2e-5, atol=1e-7)
    unrounded = normalized / scale.unsqueeze(-1)
    quant = stored[..., :16384].contiguous().view(torch.int8).view_as(state)
    # Avoid half-integer ties sensitive to FP32 reduction order. All other
    # entries must match direct FP32 RNE, including values near FP16 ties.
    stable = (unrounded.frac().abs() - 0.5).abs() > 1e-3
    torch.testing.assert_close(quant[stable], unrounded.round().to(torch.int8)[stable], rtol=0, atol=0)


def test_native_warm_fit_reuses_previous_value_factor(native):
    torch.manual_seed(411)
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.zeros(1, dtype=torch.int32, device="npu")
    native.encode_state(torch.randn(1, 32, 128, 128, device="npu") * 0.02, packet, indices)
    stored = packet.cpu()
    left = stored[..., 17408:18432].contiguous().view(torch.float16).float().view(1, 32, 4, 128)
    right = stored[..., 18432:19456].contiguous().view(torch.float16).float().view(1, 32, 4, 128)
    smooth = stored[..., 16896:17408].contiguous().view(torch.float32)
    scale = stored[..., 16384:16896].contiguous().view(torch.float32)
    state = stored[..., :16384].contiguous().view(torch.int8).float().view(1, 32, 128, 128)
    state = state * smooth.unsqueeze(-2) * scale.unsqueeze(-1) + left.transpose(-1, -2) @ right
    direction = state.transpose(-1, -2) @ left[..., 0, :].unsqueeze(-1)
    direction /= direction.square().sum(-2, keepdim=True).sqrt()
    q, k, v, beta, g = _inputs(1)
    beta.zero_()
    g.zero_()
    for _ in range(16):
        native.decode(q, k, v, beta, g, packet, indices)
    actual = packet[..., 18432:18688].contiguous().view(torch.float16).float().cpu()
    torch.testing.assert_close(actual, direction.squeeze(-1).half().float(), rtol=0.003, atol=2e-4)


def test_native_mixed_window_cursors(native):
    torch.manual_seed(412)
    packet = torch.zeros(3, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.arange(3, dtype=torch.int32, device="npu")
    native.encode_state(torch.randn(3, 32, 128, 128, device="npu") * 0.02, packet, indices)
    inputs = _inputs(3)
    for row, count in enumerate((0, 7, 15)):
        for _ in range(count):
            native.decode(*(t[row : row + 1] for t in inputs), packet, indices[row : row + 1])
    separate = packet.clone()
    for _ in range(19):
        actual = native.decode(*inputs, packet, indices)
        expected = torch.cat(
            [native.decode(*(t[row : row + 1] for t in inputs), separate, indices[row : row + 1]) for row in range(3)]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(packet, separate, rtol=0, atol=0)


def test_native_current_update_is_not_rounded_to_fp16(native):
    torch.manual_seed(413)
    q = torch.zeros(1, 16, 128, dtype=torch.bfloat16, device="npu")
    q[..., 0] = 1
    v = torch.randn(1, 32, 128).bfloat16()
    beta = torch.rand(1, 32).bfloat16()
    delta = v.float() * beta.float().unsqueeze(-1)
    expected = (delta * 128**-0.5).bfloat16()
    rounded_first = (delta.half().float() * 128**-0.5).bfloat16()
    assert (expected != rounded_first).any()
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.zeros(1, dtype=torch.int32, device="npu")
    native.encode_state(torch.zeros(1, 32, 128, 128, device="npu"), packet, indices)
    actual = native.decode(q, q, v.npu(), beta.npu(), torch.zeros(1, 32, device="npu"), packet, indices)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_native_clamps_only_buffered_update(native):
    q = torch.zeros(1, 16, 128, dtype=torch.bfloat16, device="npu")
    q[..., 0] = 1
    v = torch.full((1, 32, 128), 70000, dtype=torch.bfloat16, device="npu")
    beta = torch.ones(1, 32, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(1, 32, device="npu")
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.zeros(1, dtype=torch.int32, device="npu")
    native.encode_state(torch.zeros(1, 32, 128, 128, device="npu"), packet, indices)
    actual = native.decode(q, q, v, beta, g, packet, indices)
    torch.testing.assert_close(actual, (v.float() * 128**-0.5).bfloat16(), rtol=0, atol=0)
    buffered = packet[..., 23552:23808].contiguous().view(torch.float16).cpu()
    torch.testing.assert_close(buffered, torch.full_like(buffered, 65504), rtol=0, atol=0)
    beta.zero_()
    next_output = native.decode(q, q, v, beta, g, packet, indices)
    expected = torch.full_like(v, 65504 * 128**-0.5)
    torch.testing.assert_close(next_output, expected, rtol=0, atol=0)


def test_native_sparse_batch_matches_independent_slots(native):
    # Many jobs per core exercise asynchronous prefetch and tile ownership;
    # independent slots must match even though they run in a different order.
    torch.manual_seed(414)
    batch = 64
    rows = [0, 39, 40, 63]
    packet = torch.zeros(batch + len(rows), 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = (torch.arange(batch, dtype=torch.int32, device="npu") * 5) % batch
    native.encode_state(torch.zeros(batch, 32, 128, 128, device="npu"), packet, indices)
    shadows = torch.arange(batch, batch + len(rows), dtype=torch.int32, device="npu")
    for i, row in enumerate(rows):
        packet[batch + i].copy_(packet[indices[row]])
    inputs = _inputs(batch)
    for _ in range(17):
        actual = native.decode(*inputs, packet, indices)
        for i, row in enumerate(rows):
            expected = native.decode(*(t[row : row + 1] for t in inputs), packet, shadows[i : i + 1])
            torch.testing.assert_close(actual[row : row + 1], expected, rtol=0, atol=0)
    for i, row in enumerate(rows):
        torch.testing.assert_close(packet[indices[row]], packet[batch + i], rtol=0, atol=0)


def test_native_four_digit_residual_contraction(native_library):
    torch.npu.set_device(0)
    backend = NativeLeapQuant(str(native_library))
    torch.manual_seed(416)
    query = torch.randn(5, 2, 128) * 0.1
    query[0].zero_()
    query[1] *= 1e-20
    state = torch.randint(-127, 128, (5, 128, 128), dtype=torch.int8)
    query_device = query.npu()
    packed, scales = backend.pack_query_digits(query_device)
    assert (packed[:, 8:] == 0).all()
    assert (packed[0] == 0).all()
    products = (packed.cpu().double() @ state.double().transpose(-1, -2)).to(torch.int32).npu()
    actual = backend.unpack_query_products(products, scales)
    expected = query @ state.float().transpose(-1, -2)
    relative = ((actual.cpu() - expected).square().mean() / expected.square().mean()).sqrt()
    assert torch.isfinite(actual).all() and relative < 1e-5
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        packed_out, scales_out = backend.pack_query_digits(query_device)
        output = backend.unpack_query_products(products, scales_out)
    graph.replay()
    torch.testing.assert_close(output, actual, rtol=0, atol=0)
    with pytest.raises(ValueError, match="FP32"):
        backend.pack_query_digits(query.half().npu())
    with pytest.raises(ValueError, match="matching"):
        backend.unpack_query_products(products, scales[:, :2])


def test_native_lossless_residual_transpose(native_library):
    torch.npu.set_device(0)
    backend = NativeLeapQuant(str(native_library))
    torch.manual_seed(417)
    packets = torch.randint(-128, 128, (7, PACKET_BYTES), dtype=torch.int8, device="npu")
    source = packets[:, : 128 * 128].view(7, 128, 128)
    expected = source.transpose(-1, -2).contiguous()
    actual = backend.transpose_residual(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = backend.transpose_residual(source)
    source.copy_(source.flip(-1))
    graph.replay()
    torch.testing.assert_close(out, source.transpose(-1, -2).contiguous(), rtol=0, atol=0)


def test_cube_gdn_matches_fused_recurrence(native_library):
    torch.npu.set_device(0)
    torch.manual_seed(418)
    backend = NativeLeapQuant(str(native_library))
    state = torch.randn(2, 32, 128, 128, device="npu") * 0.02
    cube = NativeCubeGDN(backend, state)
    reference = cube.packet.clone()
    for step in range(65):
        inputs = _inputs()
        expected = backend.decode(*inputs, reference, cube.indices)
        actual = cube.step(*inputs)
        error = ((actual.float() - expected.float()).square().mean() / expected.float().square().mean()).sqrt()
        assert torch.isfinite(actual).all() and error < 0.02
        if step % 16 == 15:
            torch.testing.assert_close(cube.weights, backend.transpose_residual(cube.residual_view()), rtol=0, atol=0)


def test_cube_gdn_graph_and_checkpoint(native_library):
    torch.npu.set_device(0)
    torch.manual_seed(419)
    backend = NativeLeapQuant(str(native_library))
    cube = NativeCubeGDN(backend, torch.zeros(2, 32, 128, 128, device="npu"))
    inputs = _inputs()
    for _ in range(7):
        cube.step(*inputs)
    packet, weights = cube.packet.clone(), cube.weights.clone()
    expected = [cube.step(*inputs).clone() for _ in range(19)]
    expected_packet, expected_weights = cube.packet.clone(), cube.weights.clone()
    cube.packet.copy_(packet)
    cube.weights.copy_(weights)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = cube.step(*inputs)
    cube.packet.copy_(packet)
    cube.weights.copy_(weights)
    for result in expected:
        graph.replay()
        torch.testing.assert_close(output, result, rtol=0, atol=0)
    torch.testing.assert_close(cube.packet, expected_packet, rtol=0, atol=0)
    torch.testing.assert_close(cube.weights, expected_weights, rtol=0, atol=0)


def test_cube_gdn_free_layout_counterfactual_stays_correct(native_library):
    torch.npu.set_device(0)
    torch.manual_seed(420)
    backend = NativeLeapQuant(str(native_library))
    state = torch.randn(1, 32, 128, 128, device="npu") * 0.02
    full = NativeCubeGDN(backend, state)
    diagnostic = NativeCubeGDN(backend, state)
    inputs = _inputs(1)
    for _ in range(3):
        for _ in range(16):
            expected = full.step(*inputs)
            actual = diagnostic._step(*inputs, refresh_layout=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        diagnostic.refresh_layout()
        torch.testing.assert_close(diagnostic.packet, full.packet, rtol=0, atol=0)
        torch.testing.assert_close(diagnostic.weights, full.weights, rtol=0, atol=0)


def test_cube_gdn_window_graph_counterfactual(native_library):
    torch.npu.set_device(0)
    torch.manual_seed(421)
    backend = NativeLeapQuant(str(native_library))
    initial = torch.randn(1, 32, 128, 128, device="npu") * 0.02
    full, diagnostic = NativeCubeGDN(backend, initial), NativeCubeGDN(backend, initial)
    inputs = _inputs(1)
    full_graph, diagnostic_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(full_graph):
        for _ in range(16):
            expected = full.step(*inputs)
    with torch.npu.graph(diagnostic_graph):
        for _ in range(16):
            actual = diagnostic._step(*inputs, refresh_layout=False)
    torch.npu.synchronize()
    diagnostic.refresh_layout()
    for _ in range(3):
        full_graph.replay()
        diagnostic_graph.replay()
        torch.npu.synchronize()
        diagnostic.refresh_layout()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(diagnostic.packet, full.packet, rtol=0, atol=0)
        torch.testing.assert_close(diagnostic.weights, full.weights, rtol=0, atol=0)

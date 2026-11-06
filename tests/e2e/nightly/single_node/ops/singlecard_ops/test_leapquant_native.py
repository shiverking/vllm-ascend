# SPDX-License-Identifier: Apache-2.0
"""Isolated native library tests. Build in pytest's temporary directory only."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.leapquant_native import PACKET_BYTES, NativeLeapQuant


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    source = Path(__file__).resolve().parents[6] / "csrc/experimental/leapquant"
    build = tmp_path_factory.mktemp("leapquant_native")
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cmake", "--build", str(build), "-j4"], check=True, capture_output=True)
    torch.npu.set_device(0)
    return NativeLeapQuant(str(build / "lib/libleapquant_native.so"))


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
    packet.zero_()  # Missing LQ02 tag: no cache mutation, nonfinite output.
    actual = native.decode(q, q, v, beta, g, packet, indices)
    assert torch.isnan(actual).all()
    torch.testing.assert_close(packet, torch.zeros_like(packet), rtol=0, atol=0)


def test_native_rejects_invalid_metadata(native):
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8)
    indices = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="FP32"):
        native.encode_state(torch.zeros(1, 32, 128, 128, dtype=torch.bfloat16), packet, indices)
    with pytest.raises(ValueError, match="cores"):
        NativeLeapQuant("not-loaded", cores=0)

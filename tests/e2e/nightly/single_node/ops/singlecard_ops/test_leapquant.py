"""Device checks independent of the vLLM server; requires physical NPU access."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.leapquant import PACKET_BYTES, decode, encode_state


def test_snapshot_skip_padding_and_graph():
    torch.npu.set_device(0)
    torch.manual_seed(93)
    packet = torch.zeros(4, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    encode_state(torch.zeros(2, 32, 128, 128, device="npu"), packet, indices)
    q = torch.nn.functional.normalize(torch.randn(2, 16, 128), dim=-1).bfloat16().npu()
    k = torch.nn.functional.normalize(torch.randn(2, 16, 128), dim=-1).bfloat16().npu()
    v = torch.randn(2, 32, 128).bfloat16().npu()
    beta = torch.rand(2, 32, device="npu").bfloat16()
    g = -torch.rand(2, 32, device="npu") * 0.2
    scratch = torch.empty(2, 32, 128, 128, device="npu")

    def step():
        return decode(q, k, v, beta, g, packet, indices, scratch)

    for _ in range(5):
        step()
    packet[2].copy_(packet[1])
    saved = packet.clone()
    # A frozen prefix checkpoint includes the five pending record updates.
    packet[3].copy_(packet[2])
    for tensor in (q, k, v, beta, g):
        tensor[1].copy_(tensor[0])
    indices[1] = 3
    for _ in range(28):
        output = step()
        torch.testing.assert_close(output[0], output[1], rtol=0, atol=0)
    # The immutable source checkpoint saved before branching is unchanged.
    torch.testing.assert_close(packet[2], saved[2], rtol=0, atol=0)
    # Replay the identical captured graph across arbitrary window positions.
    before = packet.clone()
    eager = []
    for _ in range(19):
        eager.append(step().clone())
    expected = packet.clone()
    packet.copy_(before)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = step()
    packet.copy_(before)
    for expected_output in eager:
        graph.replay()
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(packet, expected, rtol=0, atol=0)
    indices.fill_(-1)
    before = packet.clone()
    graph.replay()
    assert output.abs().max().item() == 0
    torch.testing.assert_close(packet, before, rtol=0, atol=0)


def test_invalid_packet_shape_rejected_before_launch():
    packet = torch.zeros(1, 32, PACKET_BYTES - 1, dtype=torch.uint8)
    state = torch.zeros(1, 32, 128, 128)
    indices = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="packet"):
        encode_state(state, packet, indices)


def test_invalid_state_dtype_rejected_before_launch():
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8)
    state = torch.zeros(1, 32, 128, 128, dtype=torch.bfloat16)
    indices = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="FP32"):
        encode_state(state, packet, indices)


def test_low_rank_state_stays_finite():
    torch.npu.set_device(0)
    packet = torch.zeros(1, 32, PACKET_BYTES, dtype=torch.uint8, device="npu")
    indices = torch.zeros(1, dtype=torch.int32, device="npu")
    encode_state(torch.full((1, 32, 128, 128), 0.01, device="npu"), packet, indices)
    q = torch.full((1, 16, 128), 128**-0.5, dtype=torch.bfloat16, device="npu")
    v = torch.zeros(1, 32, 128, dtype=torch.bfloat16, device="npu")
    beta = torch.zeros(1, 32, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(1, 32, device="npu")
    output = decode(q, q, v, beta, g, packet, indices)
    torch.testing.assert_close(output.float(), torch.full_like(output.float(), 0.01), rtol=0, atol=2e-4)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Native asynchronous MLP must record kernels on the PyTorch capture stream."""

import pytest
import torch
import torch_npu


@pytest.mark.parametrize("batch", [1, 32, 192])
def test_xlite_mlp_changed_input_graph_replay(batch):
    xlite = pytest.importorskip("xlite._C")
    from vllm_ascend.xlite.stream_bridge import load_stream_bridge

    torch.npu.set_device(0)
    bridge = load_stream_bridge()
    runtime = xlite.Runtime(0, 32)
    x = torch.randn(batch, 128, device="npu", dtype=torch.bfloat16)
    up_weight = torch.randn(256, 128, device=x.device, dtype=x.dtype)
    down_weight = torch.randn(128, 128, device=x.device, dtype=x.dtype)
    up = torch.empty(batch, 256, device=x.device, dtype=x.dtype)
    act = torch.empty_like(x)
    out = torch.empty_like(x)

    def run():
        bridge.mlp(
            runtime, x, up_weight, down_weight, up, act, out, False, False, torch.npu.current_stream().npu_stream
        )

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # Successful capture alone is insufficient: stale output must not survive.
    x.zero_()
    out.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    assert bool(torch.isfinite(out).all())
    assert float(out.abs().max()) == 0
    x.normal_()
    graph.replay()
    torch.npu.synchronize()
    projected = torch.nn.functional.linear(x, up_weight)
    reference = torch.nn.functional.linear(torch_npu.npu_swiglu(projected), down_weight)
    relative_error = (out.float() - reference.float()).norm() / reference.float().norm()
    assert float(relative_error) < 0.02

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_ascend.xlite.qwen35_decode import DecodeWorkspace, XliteDecodeMLP, qwen35_xlite_mlp


class DenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 12, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        up = self.gate_up_proj(x)
        return self.down_proj(torch.nn.functional.silu(up[:, :6]) * up[:, 6:])


class NativeOps:
    @staticmethod
    def matmul(runtime, x, weight, out, nz, transpose):
        out.copy_(torch.nn.functional.linear(x, weight))

    @staticmethod
    def silu_and_mul(runtime, up, out):
        half = up.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(up[:, :half]) * up[:, half:])


def test_decode_zero_copy_weights_and_distinct_layer_outputs():
    w = DecodeWorkspace(None, NativeOps, torch.empty(8, 12), torch.empty(8, 6))
    original = DenseMLP()
    with patch("torch_npu.get_npu_format", return_value=0):
        a = XliteDecodeMLP(original, w, 8, "a")
        b = XliteDecodeMLP(DenseMLP(), w, 8, "b")
    x = torch.randn(2, 4)
    with (
        patch(
            "vllm_ascend.xlite.qwen35_decode.get_forward_context", return_value=NS(no_compile_layers={"a": a, "b": b})
        ),
        patch("torch.ops.vllm.qwen35_xlite_mlp", qwen35_xlite_mlp),
    ):
        torch.testing.assert_close(a(x), original(x))  # Prefill fallback.
        w.enabled = True
        output = a(x)
        reference = output.clone()
        torch.testing.assert_close(output, original(x))
        b(x)
        torch.testing.assert_close(output, reference)
    assert a.original.gate_up_proj.weight.data_ptr() == original.gate_up_proj.weight.data_ptr()
    assert a.output.data_ptr() != b.output.data_ptr()
    assert output.data_ptr() == a.output.data_ptr()
    assert w.native_python_calls == 2
    with (
        patch("vllm_ascend.xlite.qwen35_decode.get_forward_context", return_value=NS(no_compile_layers={"a": a})),
        patch("torch.ops.vllm.qwen35_xlite_mlp", qwen35_xlite_mlp),
        pytest.raises(RuntimeError, match="capacity"),
    ):
        a(torch.randn(9, 4))


def test_router_preserves_prefill_and_resets_after_failure():
    from vllm_ascend.xlite.qwen35_decode import AscendAttentionState, Qwen35DecodeMLPWrapper

    wrapper = object.__new__(Qwen35DecodeMLPWrapper)
    wrapper.workspace = NS(enabled=False)
    wrapper._routing_logged = set()
    seen = []
    wrapper.runnable = lambda *args, **kwargs: seen.append(wrapper.workspace.enabled)

    class Metadata:
        def __init__(self, state):
            self.attn_state = state

    module = "vllm_ascend.xlite.qwen35_decode"
    with patch(module + ".AscendMetadata", Metadata):
        for metadata, expected in [
            (None, False),
            ({"gdn_first": object(), "mha": Metadata(AscendAttentionState.DecodeOnly)}, True),
            ({"gdn_first": object()}, False),
        ]:
            with patch(module + ".get_forward_context", return_value=NS(attn_metadata=metadata)):
                wrapper()
                assert seen[-1] == expected
                assert not wrapper.workspace.enabled
        wrapper.runnable = lambda: (_ for _ in ()).throw(RuntimeError("test"))
        with patch(
            module + ".get_forward_context",
            return_value=NS(attn_metadata={"mha": Metadata(AscendAttentionState.DecodeOnly)}),
        ):
            with pytest.raises(RuntimeError, match="test"):
                wrapper()
            assert not wrapper.workspace.enabled

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Decode MLP-only xLite integration with no recurrent-state staging."""

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

RUNTIME_POOL_MB = 32
DIAGNOSTIC_INTERVAL = 128


@dataclass
class DecodeWorkspace:
    runtime: Any
    ops: Any
    up_gate: torch.Tensor
    activated: torch.Tensor
    enabled: bool = False
    native_python_calls: int = 0
    stream_bridge: Any = None


class XliteDecodeMLP(nn.Module):
    """Reuse original weights and shared activation scratch; never touch KV."""

    def __init__(self, original, workspace, max_tokens, layer_name):
        super().__init__()
        self.original = original
        self.layer_name = layer_name
        self.workspace = workspace
        weight = original.down_proj.weight
        self.output = torch.empty((max_tokens, weight.shape[0]), dtype=weight.dtype, device=weight.device)
        self.up_nz = torch_npu.get_npu_format(original.gate_up_proj.weight) == torch_npu.Format.FRACTAL_NZ
        self.down_nz = torch_npu.get_npu_format(weight) == torch_npu.Format.FRACTAL_NZ

    def forward(self, hidden_states):
        # Routing must stay INSIDE this opaque op: torch.compile otherwise
        # specializes the Prefill warmup's Python flag and bypasses xLite forever.
        return torch.ops.vllm.qwen35_xlite_mlp(hidden_states, self.layer_name)


class Qwen35DecodeMLPWrapper:
    """Preserve the original multimodal/GDN execution and replace only Decode MLPs."""

    def __init__(self, runnable, config, device):
        # Native extension import stays worker-local.
        from xlite import _C

        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.xlite.stream_bridge import load_stream_bridge

        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or config.speculative_config is not None
            or config.quant_config is not None
            or config.lora_config is not None
            or config.model_config.dtype != torch.bfloat16
            or get_ascend_config().xlite_graph_config.full_mode
        ):
            raise ValueError(
                "Qwen3.5 xLite MLP requires BF16 TP=PP=DP=1, decode-only, no quantization/speculation/LoRA"
            )
        self.runnable = runnable
        options = config.additional_config.get("xlite_graph_config", {})
        self.reference = bool(options.get("qwen35_decode_mlp_reference", False))
        self.diagnostics = bool(options.get("qwen35_decode_mlp_diagnostics", False))
        self.route_counts = Counter()
        self.decode_tokens = Counter()
        self.diagnostic_steps = 0
        self._routing_logged = set()
        text = config.model_config.hf_text_config
        maximum = config.scheduler_config.max_num_seqs
        runtime = _C.Runtime(devid=device.index or 0, size=RUNTIME_POOL_MB)
        self.workspace = DecodeWorkspace(
            runtime,
            _C,
            torch.empty((maximum, 2 * text.intermediate_size), dtype=torch.bfloat16, device=device),
            torch.empty((maximum, text.intermediate_size), dtype=torch.bfloat16, device=device),
            stream_bridge=load_stream_bridge(),
        )
        layers = runnable.language_model.model.layers
        for index, layer in enumerate(layers):
            mlp = layer.mlp
            if getattr(mlp, "expert_gate", None) is not None:
                raise ValueError("Qwen3.5 xLite MLP does not support expert gates")
            name = f"qwen35_xlite_decode_mlp_{index}"
            context = config.compilation_config.static_forward_context
            if name in context:
                raise ValueError(f"Duplicate xLite layer registration: {name}")
            layer.mlp = XliteDecodeMLP(mlp, self.workspace, maximum, name)
            context[name] = layer.mlp
        logger.info(
            "Qwen3.5 xLite MLP weights: up_format=%s down_format=%s reference=%s diagnostics=%s",
            torch_npu.get_npu_format(layers[0].mlp.original.gate_up_proj.weight),
            torch_npu.get_npu_format(layers[0].mlp.original.down_proj.weight),
            self.reference,
            self.diagnostics,
        )
        logger.info(
            "Qwen3.5 xLite Decode MLP enabled for %d layers; KV/GDN state staging=0 bytes; "
            "native calls during graph capture are not replay counts",
            len(layers),
        )

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "runnable"), name)

    def unwrap(self):
        runnable = self.runnable
        return runnable.unwrap() if isinstance(runnable, ACLGraphWrapper) else runnable

    def register_kv_caches(self, kv_caches):
        # All attention/recurrent operators keep their original cache bindings.
        pass

    def __call__(self, *args, **kwargs):
        context = get_forward_context()
        metadata = context.attn_metadata
        metadata = metadata[0] if isinstance(metadata, list) else metadata
        values = metadata.values() if isinstance(metadata, dict) else ()
        mha = next((m for m in values if isinstance(m, AscendMetadata)), None)
        decode_only = mha is not None and mha.attn_state == AscendAttentionState.DecodeOnly
        self.workspace.enabled = decode_only and not getattr(self, "reference", False)
        if getattr(self, "diagnostics", False) and mha is not None and not context.capturing:
            descriptor = context.batch_descriptor
            padded = descriptor.num_tokens if descriptor is not None else 0
            key = f"{mha.attn_state.name}:padded={padded}:graph={context.cudagraph_runtime_mode.name}"
            self.route_counts[key] += 1
            self.decode_tokens[mha.attn_state.name] += mha.num_decode_tokens
            self.diagnostic_steps += 1
            if self.diagnostic_steps % DIAGNOSTIC_INTERVAL == 0:
                logger.info(
                    "Qwen3.5 xLite route totals: %s",
                    json.dumps(
                        {
                            "steps": self.diagnostic_steps,
                            "routes": self.route_counts,
                            "decode_tokens": self.decode_tokens,
                        },
                        sort_keys=True,
                    ),
                )
        route = (type(mha).__name__, str(getattr(mha, "attn_state", None)))
        if route not in self._routing_logged:
            self._routing_logged.add(route)
            logger.info("Qwen3.5 xLite MLP routing: metadata=%s state=%s enabled=%s", *route, self.workspace.enabled)
        try:
            return self.runnable(*args, **kwargs)
        finally:
            self.workspace.enabled = False


def qwen35_xlite_mlp(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    w = layer.workspace
    if not w.enabled:
        return layer.original(x)
    count = x.shape[0]
    if count > layer.output.shape[0]:
        raise RuntimeError("xLite Decode MLP exceeded preallocated token capacity")
    up, act, out = w.up_gate[:count], w.activated[:count], layer.output[:count]
    up_weight = layer.original.gate_up_proj.weight
    down_weight = layer.original.down_proj.weight
    if w.stream_bridge is not None:
        w.stream_bridge.mlp(
            w.runtime,
            x,
            up_weight,
            down_weight,
            up,
            act,
            out,
            layer.up_nz,
            layer.down_nz,
            torch.npu.current_stream().npu_stream,
        )
    else:
        # CPU test double; production always supplies the asynchronous bridge.
        w.ops.matmul(w.runtime, x, up_weight, up, layer.up_nz, False)
        w.ops.silu_and_mul(w.runtime, up, act)
        w.ops.matmul(w.runtime, act, down_weight, out, layer.down_nz, False)
    w.native_python_calls += 1
    return out


def qwen35_xlite_mlp_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="qwen35_xlite_mlp",
    op_func=qwen35_xlite_mlp,
    mutates_args=[],
    fake_impl=qwen35_xlite_mlp_fake,
)

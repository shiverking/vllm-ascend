# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.leapquant_config import LeapQuantConfig


def config(path):
    return SimpleNamespace(
        additional_config={"leapquant_config": {"enabled": True, "library": str(path)}},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5"),
            hf_text_config=SimpleNamespace(
                linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128
            ),
            dtype=torch.bfloat16,
            quantization=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=None,
        kv_transfer_config=None,
    )


def test_leapquant_serving_validation(tmp_path):
    path = tmp_path / "library.so"
    path.touch()
    cfg = config(path)
    assert LeapQuantConfig.from_vllm_config(cfg).enabled
    cfg.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="TP/PP"):
        LeapQuantConfig.from_vllm_config(cfg)
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.speculative_config = object()
    with pytest.raises(ValueError, match="MTP"):
        LeapQuantConfig.from_vllm_config(cfg)
    cfg.speculative_config = None
    cfg.cache_config.cache_dtype = "fp8"
    with pytest.raises(ValueError, match="quantization"):
        LeapQuantConfig.from_vllm_config(cfg)
    cfg.cache_config.cache_dtype = "auto"
    cfg.additional_config["leapquant_config"]["library"] = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="library"):
        LeapQuantConfig.from_vllm_config(cfg)
    cfg.additional_config["leapquant_config"]["library"] = str(path)
    cfg.cache_config.user_specified_block_size = True
    cfg.cache_config.block_size = 1024
    with pytest.raises(ValueError, match="512-token"):
        LeapQuantConfig.from_vllm_config(cfg)
    cfg.cache_config.block_size = 512
    assert LeapQuantConfig.from_vllm_config(cfg).enabled
    cfg.additional_config = {}
    assert not LeapQuantConfig.from_vllm_config(cfg).enabled

# SPDX-License-Identifier: Apache-2.0
"""Validated, opt-in serving configuration for the TP1 LeapQuant experiment."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class LeapQuantConfig:
    enabled: bool = False
    library: str = ""
    cores: int = 40

    @classmethod
    def from_vllm_config(cls, config):
        options = (config.additional_config or {}).get("leapquant_config", {})
        if not isinstance(options, dict):
            raise ValueError("leapquant_config must be a dictionary")
        if not options.get("enabled", False):
            return cls()
        unknown = set(options) - {"enabled", "library", "cores"}
        if unknown:
            raise ValueError(f"Unknown leapquant_config keys: {sorted(unknown)}")
        model = config.model_config
        parallel = config.parallel_config
        hf = model.hf_text_config
        if model.hf_config.model_type != "qwen3_5" or (
            hf.linear_num_key_heads,
            hf.linear_num_value_heads,
            hf.linear_key_head_dim,
            hf.linear_value_head_dim,
        ) != (16, 32, 128, 128):
            raise ValueError("LeapQuant serving requires Qwen3.5-9B GDN dimensions")
        if model.dtype != torch.bfloat16 or any(
            getattr(parallel, key, 1) != 1
            for key in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ):
            raise ValueError("LeapQuant serving requires BF16 and TP/PP/PCP/DCP=1")
        if config.speculative_config is not None or config.kv_transfer_config is not None:
            raise ValueError("LeapQuant serving does not support MTP or cache transfer")
        if model.quantization is not None or config.cache_config.cache_dtype != "auto":
            raise ValueError("LeapQuant serving excludes weight and full-attention KV quantization")
        cache = config.cache_config
        for field in ("block_size", "mamba_block_size"):
            if getattr(cache, f"user_specified_{field}", False) and getattr(cache, field) != 512:
                raise ValueError("LeapQuant serving requires 512-token attention/Mamba blocks")
        cores = options.get("cores", 40)
        if not isinstance(cores, int) or not 1 <= cores <= 48:
            raise ValueError("LeapQuant cores must be in [1,48]")
        library = Path(options.get("library", ""))
        if not library.is_file():
            raise ValueError("LeapQuant requires an existing native library")
        return cls(True, str(library.resolve()), cores)

# SPDX-License-Identifier: Apache-2.0
"""Opt-in, independently measurable data movement optimizations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataMovementConfig:
    prefill_state_layout: bool = False
    batched_prefix_copy: bool = False
    greedy_no_fp32_cast: bool = False

    @classmethod
    def from_config(cls, config):
        options = (getattr(config, "additional_config", None) or {}).get("data_movement_optimizations", {})
        if not isinstance(options, dict) or set(options) - set(cls.__dataclass_fields__):
            raise ValueError("Invalid data_movement_optimizations keys")
        if any(type(value) is not bool for value in options.values()):
            raise ValueError("data_movement_optimizations values must be booleans")
        if any(options.values()) and any(
            getattr(config.parallel_config, key, 1) != 1
            for key in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ):
            raise ValueError("Data movement optimizations require TP/PP/PCP/DCP=1")
        return cls(**options)

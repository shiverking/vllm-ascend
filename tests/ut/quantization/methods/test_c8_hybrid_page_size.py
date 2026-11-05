"""Regression checks for early pure-C8 hybrid cache page sizing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase


class TestC8HybridPageSize(TestBase):
    def _run(self, layout, covered=True):
        from vllm_ascend.patch.platform.patch_mamba_config import verify_and_update_config

        with TemporaryDirectory() as directory:
            desc = {"kv_cache_type": "C8"}
            if covered:
                desc["model.layers.1.self_attn.k_proj.kv_cache_scale"] = "FLOAT"
            (Path(directory) / "quant_model_description.json").write_text(json.dumps(desc))
            cache = SimpleNamespace(
                cache_dtype="auto",
                block_size=None,
                mamba_page_size_padded=None,
                enable_prefix_caching=True,
                mamba_cache_mode="align",
            )
            config = SimpleNamespace(
                cache_config=cache,
                quant_config=None,
                kv_transfer_config=None,
                speculative_config=None,
                additional_config={"c8_kv_cache_layout": layout},
                parallel_config=None,
                scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
                model_config=SimpleNamespace(
                    model=directory,
                    architecture="Qwen3_5ForConditionalGeneration",
                    dtype=torch.bfloat16,
                    use_mla=False,
                    hf_text_config=SimpleNamespace(layer_types=["linear_attention", "full_attention"]),
                    get_num_kv_heads=lambda _: 4,
                    get_head_size=lambda: 256,
                    max_model_len=16384,
                ),
            )
            model_class = SimpleNamespace(
                get_mamba_state_shape_from_config=lambda _: [(4, 6144), (32, 128, 128)],
                get_mamba_state_dtype_from_config=lambda _: [torch.bfloat16, torch.float32],
            )
            with (
                patch("vllm_ascend.patch.platform.patch_mamba_config.MambaModelConfig.verify_and_update_config"),
                patch(
                    "vllm_ascend.patch.platform.patch_mamba_config.ModelRegistry.resolve_model_cls",
                    return_value=(model_class, None),
                ),
            ):
                verify_and_update_config.__func__(None, config)
            return cache

    def test_default_bf16_page_size_unchanged(self):
        self.assertEqual(self._run("NZ").block_size, 1024)

    def test_c8_uses_actual_dtype_before_quant_config_is_constructed(self):
        cache = self._run("ND")
        self.assertEqual(cache.block_size, 2048)
        self.assertEqual(cache.mamba_block_size, 2048)

    def test_partial_c8_hybrid_rejected(self):
        with self.assertRaisesRegex(ValueError, "every full-attention"):
            self._run("ND", covered=False)

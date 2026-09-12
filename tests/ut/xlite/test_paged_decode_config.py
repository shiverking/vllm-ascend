"""Dependency-light checks of the actual configuration class (no NPU import)."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


class PagedDecodeConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend" / "ascend_config.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "XliteGraphConfig")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cls.config_type = namespace["XliteGraphConfig"]

    def config(self, values):
        return self.config_type(values, SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            cache_config=SimpleNamespace(block_size=128),
        ))

    def test_legacy_default(self):
        self.assertEqual(self.config({}).decode_attention_backend, "legacy")
        self.assertEqual(self.config({}).prefill_attention_backend, "legacy")
        self.assertEqual(self.config({}).matmul_optimization, "legacy")
        self.assertEqual(self.config({}).matmul_backend, "m200_asr")

    def test_matmul_backend(self):
        self.assertEqual(self.config({"matmul_backend": "aclnn"}).matmul_backend, "aclnn")
        self.assertEqual(
            self.config({"matmul_backend": "m200_asr_prefill"}).matmul_backend,
            "m200_asr_prefill")
        with self.assertRaises(ValueError):
            self.config({"matmul_backend": "unknown"})

    def test_decode_graph_is_explicit_and_strict(self):
        self.assertFalse(self.config({}).decode_graph)
        config = self.config({
            "enabled": True,
            "full_mode": True,
            "decode_attention_backend": "direct_atb",
            "direct_atb_setup_reuse": True,
            "matmul_backend": "m200_asr",
            "decode_graph": True,
        })
        self.assertTrue(config.decode_graph)
        for values in (
            {"decode_graph": True},
            {"enabled": True, "full_mode": True, "decode_graph": True},
            {"enabled": True, "full_mode": True,
             "decode_attention_backend": "direct_atb", "decode_graph": True},
            {"enabled": True, "full_mode": True,
             "decode_attention_backend": "direct_atb",
             "direct_atb_setup_reuse": True,
             "matmul_backend": "m200_asr_prefill", "decode_graph": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)

    def test_aclnn_matmul_async_requires_full_mode(self):
        config = self.config({"enabled": True, "full_mode": True,
                              "aclnn_matmul_async": True})
        self.assertTrue(config.aclnn_matmul_async)
        for values in ({"aclnn_matmul_async": True},
                       {"enabled": True, "aclnn_matmul_async": True},
                       {"enabled": True, "full_mode": True,
                        "aclnn_matmul_async": "yes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)

    def test_p3_config(self):
        config = self.config({"enabled": True, "full_mode": True, "matmul_optimization": "p3_aclnn"})
        self.assertEqual(config.matmul_optimization, "p3_aclnn")
        for values in ({"matmul_optimization": "bad"}, {"matmul_policy": "x.json"},
                       {"matmul_optimization": "p3_aclnn"},
                       {"enabled": True, "full_mode": True, "matmul_optimization": "p3_aclnn",
                        "decode_attention_backend": "paged_310p"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)

    def test_paged_full_mode(self):
        self.assertEqual(self.config({"enabled": True, "full_mode": True,
                                     "decode_attention_backend": "paged_310p"}).decode_attention_backend,
                         "paged_310p")

    def test_batched_aclnn_full_mode(self):
        self.assertEqual(
            self.config({"enabled": True, "full_mode": True,
                         "decode_attention_backend": "batched_aclnn"}).decode_attention_backend,
            "batched_aclnn")
        for values in ({"decode_attention_backend": "batched_aclnn"},
                       {"enabled": True, "decode_attention_backend": "batched_aclnn"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)

    def test_native_atb_full_mode(self):
        self.assertEqual(
            self.config({"enabled": True, "full_mode": True,
                         "decode_attention_backend": "native_atb"}).decode_attention_backend,
            "native_atb")

    def test_direct_atb_full_mode(self):
        self.assertEqual(
            self.config({"enabled": True, "full_mode": True,
                         "decode_attention_backend": "direct_atb"}).decode_attention_backend,
            "direct_atb")
        self.assertTrue(
            self.config({"enabled": True, "full_mode": True,
                         "decode_attention_backend": "direct_atb",
                         "direct_atb_setup_reuse": True}).direct_atb_setup_reuse)
        for values in ({"direct_atb_setup_reuse": True},
                       {"enabled": True, "full_mode": True,
                        "decode_attention_backend": "legacy",
                        "direct_atb_setup_reuse": True},
                       {"enabled": True, "full_mode": True,
                        "decode_attention_backend": "direct_atb",
                        "direct_atb_setup_reuse": "yes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)

    def test_batched_prefill_probe_requires_direct_atb_full_mode(self):
        config = self.config({
            "enabled": True,
            "full_mode": True,
            "decode_attention_backend": "direct_atb",
            "prefill_attention_backend": "batched_aclnn_probe",
        })
        self.assertEqual(config.prefill_attention_backend, "batched_aclnn_probe")
        for values in (
            {"prefill_attention_backend": "unknown"},
            {"prefill_attention_backend": "batched_aclnn_probe"},
            {"enabled": True, "full_mode": True,
             "prefill_attention_backend": "batched_aclnn_probe"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)

    def test_invalid_and_decode_only_rejected(self):
        for values in ({"decode_attention_backend": "unknown"},
                       {"decode_attention_backend": "direct_atb"},
                       {"decode_attention_backend": "native_atb"},
                       {"decode_attention_backend": "paged_310p"},
                       {"enabled": True, "decode_attention_backend": "paged_310p"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(values)

    def test_capability_check(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend" / "xlite" / "paged_decode.py"
        spec = importlib.util.spec_from_file_location("paged_check", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime = SimpleNamespace(set_decode_attention_backend=lambda backend: None)
        valid = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                 "cache_layout": "BSHD", "paged_decode_310p": True}
        module.validate_paged_decode_build("paged_310p", valid, runtime)
        for key in valid:
            bad = dict(valid)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                module.validate_paged_decode_build("paged_310p", bad, runtime)
        with self.assertRaises(RuntimeError):
            module.validate_paged_decode_build("paged_310p", valid, SimpleNamespace())
        module.validate_paged_decode_build("legacy", {}, SimpleNamespace())
        batched = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                   "cache_layout": "BSHD", "batched_decode_attention": True,
                   "decode_attention_backends": ("batched_aclnn", "legacy")}
        module.validate_paged_decode_build("batched_aclnn", batched, runtime)
        with self.assertRaises(RuntimeError):
            module.validate_paged_decode_build("batched_aclnn", {}, runtime)
        native = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                  "cache_layout": "BSHD", "native_decode_attention": True,
                  "native_decode_cache_layout": "NZ_5D",
                  "decode_attention_backends": ("native_atb", "legacy")}
        module.validate_paged_decode_build("native_atb", native, runtime)
        with self.assertRaises(RuntimeError):
            module.validate_paged_decode_build("native_atb", {}, runtime)
        direct = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                  "cache_layout": "BSHD", "direct_decode_attention": True,
                  "native_decode_cache_layout": "NZ_5D",
                  "direct_atb_task_queue_independent": True,
                  "direct_atb_runtime_version": 9,
                  "direct_atb_operation_scope": "per_layer_batch",
                  "direct_atb_setup_cache": True,
                  "direct_atb_fused_rope_staging": True,
                  "direct_atb_mixed_batch_decode": True,
                  "direct_atb_batched_compact_scatter": True,
                  "direct_atb_plan_cache": "layer_batch",
                  "direct_atb_metadata_upload": "once_per_forward",
                  "direct_atb_pure_decode_direct_output": False,
                  "direct_atb_metadata_single_h2d": False,
                  "direct_atb_metadata_actual_batch": True,
                  "direct_atb_setup_reuse_probe": True,
                  "decode_attention_backends": ("direct_atb", "legacy")}
        module.validate_paged_decode_build("direct_atb", direct, runtime)
        with self.assertRaises(RuntimeError):
            module.validate_paged_decode_build("direct_atb", {}, runtime)


if __name__ == "__main__":
    unittest.main()

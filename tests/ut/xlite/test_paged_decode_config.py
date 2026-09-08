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

    def test_paged_full_mode(self):
        self.assertEqual(self.config({"enabled": True, "full_mode": True,
                                     "decode_attention_backend": "paged_310p"}).decode_attention_backend,
                         "paged_310p")

    def test_invalid_and_decode_only_rejected(self):
        for values in ({"decode_attention_backend": "unknown"},
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


if __name__ == "__main__":
    unittest.main()

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from tests.ut.base import TestBase
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _prepare_inputs_source() -> str:
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def _prepare_inputs(")
    end = source.index("    @torch.inference_mode()", start)
    return source[start:end]


def test_prepare_inputs_keeps_aclgraph_metadata_on_cpu() -> None:
    source = _prepare_inputs_source()

    assert "block_table.compute_slot_mapping(" in source
    assert "req_indices," in source
    assert "positions_np[:total_num_scheduled_tokens]" in source

    assert "self.input_batch.block_table.compute_slot_mapping(" not in source
    assert "query_start_loc.gpu[: num_reqs + 1]" not in source
    assert "req_indices_gpu" not in source
    assert "self.num_computed_tokens[req_indices_gpu]" not in source

    assert "self.positions[:total_num_scheduled_tokens].copy_(" in source
    assert "self._positions_cpu_buf[:total_num_scheduled_tokens]" in source
    assert "self.seq_lens[:num_reqs].copy_(" in source
    assert "self.optimistic_seq_lens_cpu[:num_reqs]" in source


class TestNPUModelRunner310(TestBase):
    @staticmethod
    def _make_xlite_runner(**overrides):
        runner = object.__new__(NPUModelRunner310)
        values = {
            "dtype": torch.float16,
            "max_model_len": 2048,
            "architectures": ["Qwen3ASRForConditionalGeneration"],
        }
        values.update(overrides.pop("model_config", {}))
        runner.model_config = SimpleNamespace(**values)
        runner.cache_config = SimpleNamespace(block_size=overrides.pop("block_size", 128))
        runner.scheduler_config = SimpleNamespace(
            max_num_seqs=overrides.pop("max_num_seqs", 20),
            max_num_batched_tokens=overrides.pop("max_num_batched_tokens", 4096),
        )
        runner.ascend_config = SimpleNamespace(xlite_graph_config=SimpleNamespace(full_mode=True))
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=overrides.pop("tensor_parallel_size", 1),
                pipeline_parallel_size=overrides.pop("pipeline_parallel_size", 1),
                data_parallel_size=overrides.pop("data_parallel_size", 1),
            ),
            quant_config=overrides.pop("quant_config", None),
        )
        assert not overrides
        return runner

    def test_xlite_310p_config_accepts_poc_contract(self):
        runner = self._make_xlite_runner()
        runner._validate_xlite_310p_config()

    def test_xlite_310p_config_rejects_unsafe_modes(self):
        runner = self._make_xlite_runner(
            block_size=64,
            max_num_seqs=21,
            tensor_parallel_size=2,
            model_config={"dtype": torch.bfloat16, "max_model_len": 4096},
        )
        with self.assertRaisesRegex(ValueError, "dtype must be float16") as raised:
            runner._validate_xlite_310p_config()
        message = str(raised.exception)
        self.assertIn("tensor_parallel_size must be 1", message)
        self.assertIn("block_size must be 128", message)
        self.assertIn("max_num_seqs must be <= 20", message)
        self.assertIn("max_model_len must be <= 2048", message)

    def test_xlite_310p_allocates_fp16_bshd_nd_cache(self):
        runner = object.__new__(NPUModelRunner310)
        runner.device = torch.device("cpu")
        cache_spec = AttentionSpec(
            block_size=128,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )

        def fake_empty_with_format(*, size, dtype, device, acl_format):
            self.assertEqual(acl_format, 2)
            return torch.empty(size, dtype=dtype, device=device)

        with patch(
            "vllm_ascend._310p.model_runner_310p.torch_npu.empty_with_format",
            side_effect=fake_empty_with_format,
        ) as empty_with_format:
            key, value = runner._allocate_xlite_attention_cache(17, cache_spec)

        self.assertEqual(key.shape, (17, 128, 8, 128))
        self.assertEqual(value.shape, key.shape)
        self.assertEqual(key.dtype, torch.float16)
        self.assertEqual(empty_with_format.call_count, 2)

    def test_xlite_profile_fallback_is_scoped_to_dummy_profile(self):
        runner = object.__new__(NPUModelRunner310)
        runner._xlite_enabled = True
        runner.model = SimpleNamespace(_allow_profile_fallback=False)

        def inspect_profile_scope(*args, **kwargs):
            self.assertTrue(runner.model._allow_profile_fallback)
            return "profiled"

        with patch.object(NPUModelRunner, "_dummy_run", side_effect=inspect_profile_scope):
            result = runner._dummy_run(8, is_profile=True)

        self.assertEqual(result, "profiled")
        self.assertFalse(runner.model._allow_profile_fallback)

    def test_may_reinitialize_input_batch_expands_prefix_mamba_block_table(self):
        runner = object.__new__(NPUModelRunner310)
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.max_encoder_len = 0
        runner.max_num_tokens = 1024
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner.is_pooling_model = False
        runner.model_config = SimpleNamespace(max_model_len=512, get_vocab_size=lambda: 32000)
        runner.cache_config = SimpleNamespace(block_size=128, enable_prefix_caching=True)
        runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=4)
        runner.vllm_config = SimpleNamespace(speculative_config=None)
        runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
        runner.input_batch = SimpleNamespace(logitsprocs=MagicMock())
        attention_backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64])
        runner.attn_groups = [[SimpleNamespace(backend=attention_backend)]]

        attention_spec = AttentionSpec(
            block_size=128,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        )
        mamba_spec = MambaSpec(
            block_size=128,
            shapes=((16,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=2,
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec),
                SimpleNamespace(kv_cache_spec=mamba_spec),
            ]
        )

        with (
            patch("vllm_ascend._310p.model_runner_310p.NPUInputBatch") as mock_input_batch,
            patch("vllm_ascend._310p.model_runner_310p.get_total_cp_world_size", return_value=1),
        ):
            runner.may_reinitialize_input_batch(kv_cache_config)

        kwargs = mock_input_batch.call_args.kwargs
        self.assertEqual(kwargs["block_sizes"], [128, 128])
        self.assertEqual(kwargs["kernel_block_sizes"], [[128, 64], [0]])
        self.assertEqual(kwargs["max_num_blocks_per_req"], [4, 6])
        self.assertIs(kwargs["kv_cache_groups"], kv_cache_config.kv_cache_groups)
        self.assertEqual(kwargs["cp_kv_cache_interleave_size"], 4)

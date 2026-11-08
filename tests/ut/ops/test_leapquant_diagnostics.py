# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.leapquant_diagnostics import LeapQuantWorkerExtension


def test_diagnostics_records_loaded_library_and_state_metadata(tmp_path):
    library = tmp_path / "library.so"
    library.write_bytes(b"test library identity")
    layer = SimpleNamespace(
        leapquant=SimpleNamespace(library=SimpleNamespace(_name=str(library)), cores=40),
        kv_cache=(torch.empty(2, 32, 32768, dtype=torch.uint8),),
    )
    layers = [("model.layers.0.linear_attn", layer)]
    worker = LeapQuantWorkerExtension()
    worker.model_runner = SimpleNamespace(
        get_model=lambda: SimpleNamespace(named_modules=lambda: layers),
        kv_cache_config=SimpleNamespace(num_blocks=42),
    )
    worker.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=256),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[192, 256]),
    )
    result = worker.leapquant_cache_diagnostics()
    assert result["native_libraries_sha256"] == {str(library): hashlib.sha256(library.read_bytes()).hexdigest()}
    assert result["gdn_layers"][layers[0][0]]["tensors"][0]["logical_bytes"] == 2 * 32 * 32768
    assert result["num_gpu_blocks"] == 42
    del layer.leapquant
    result = worker.leapquant_cache_diagnostics()
    assert not result["gdn_layers"][layers[0][0]]["compressed"]
    assert result["native_libraries_sha256"] == {}


def test_cpu_batch_observer_preserves_execution_and_reports_truncation():
    worker = LeapQuantWorkerExtension()
    called = []

    def execute(output, marker=None):
        called.append(marker)
        return "original result"

    worker.model_runner = SimpleNamespace(execute_model=execute)
    worker.leapquant_start_batch_observer(max_samples=2)
    with pytest.raises(RuntimeError, match="already installed"):
        worker.leapquant_start_batch_observer()
    output = SimpleNamespace(num_scheduled_tokens={"decode": 1, "prefill": 128})
    for index in range(3):
        assert worker.model_runner.execute_model(output, marker=index) == "original result"
    observed = worker.leapquant_export_batch_observer()
    assert called == [0, 1, 2]
    assert observed["total_observed"] == 3 and observed["dropped_samples"] == 1
    assert all(row[1:] == [2, 129, 1] for row in observed["iterations"])
    worker.leapquant_reset_batch_observer()
    assert worker.leapquant_export_batch_observer()["iterations"] == []
    worker.leapquant_stop_batch_observer()
    assert worker.model_runner.execute_model is execute
    with pytest.raises(RuntimeError, match="not installed"):
        worker.leapquant_stop_batch_observer()


@pytest.mark.parametrize("size", [0, -1, 1.5, True, 1000001])
def test_observer_rejects_invalid_capacity(size):
    with pytest.raises(ValueError, match="max_samples"):
        LeapQuantWorkerExtension().leapquant_start_batch_observer(size)

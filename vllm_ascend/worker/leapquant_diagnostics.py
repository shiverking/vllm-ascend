# SPDX-License-Identifier: Apache-2.0
"""Optional LeapQuant metadata and CPU-only scheduling observer for worker RPC."""

import hashlib
import time
from collections import deque
from pathlib import Path
from types import MethodType


class LeapQuantWorkerExtension:
    """Use with --worker-extension-cls; importing this module installs no hooks."""

    def leapquant_cache_diagnostics(self):
        runner = self.model_runner
        gdn_layers, attention_layers, libraries = {}, {}, {}
        for name, layer in runner.get_model().named_modules():
            if name.endswith("linear_attn") and hasattr(layer, "kv_cache"):
                backend = getattr(layer, "leapquant", None)
                library = backend.library._name if backend is not None else None
                gdn_layers[name] = {
                    "compressed": backend is not None,
                    "native_library": library,
                    "native_cores": backend.cores if backend is not None else None,
                    "tensors": [
                        {
                            "shape": list(t.shape),
                            "dtype": str(t.dtype),
                            "stride": list(t.stride()),
                            "logical_bytes": t.numel() * t.element_size(),
                        }
                        for t in layer.kv_cache
                    ],
                }
                if library and library not in libraries:
                    libraries[library] = hashlib.sha256(Path(library).read_bytes()).hexdigest()
            elif name.endswith("self_attn.attn"):
                attention_layers[name] = {"dtype": str(layer.kv_cache_torch_dtype)}
        return {
            "gdn_layers": gdn_layers,
            "attention_layers": attention_layers,
            "native_libraries_sha256": libraries,
            "num_gpu_blocks": runner.kv_cache_config.num_blocks,
            "max_num_seqs": self.vllm_config.scheduler_config.max_num_seqs,
            "graph_capture_sizes": list(self.vllm_config.compilation_config.cudagraph_capture_sizes),
        }

    def leapquant_start_batch_observer(self, max_samples=100000):
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 1 <= max_samples <= 1000000:
            raise ValueError("max_samples must be an integer in [1,1000000]")
        if getattr(self, "_leapquant_original_execute", None) is not None:
            raise RuntimeError("LeapQuant batch observer already installed")
        self._leapquant_iterations = deque(maxlen=max_samples)
        self._leapquant_observed = 0
        original = self.model_runner.execute_model
        self._leapquant_original_execute = original

        def observe(runner, scheduler_output, *args, **kwargs):
            scheduled = scheduler_output.num_scheduled_tokens
            if scheduled:
                counts = scheduled.values()
                self._leapquant_iterations.append(
                    [time.time(), len(scheduled), sum(counts), sum(n == 1 for n in counts)]
                )
                self._leapquant_observed += 1
            return original(scheduler_output, *args, **kwargs)

        self.model_runner.execute_model = MethodType(observe, self.model_runner)
        return {"installed": True, "max_samples": max_samples}

    def leapquant_reset_batch_observer(self):
        self._leapquant_iterations.clear()
        self._leapquant_observed = 0
        return {"reset": True}

    def leapquant_export_batch_observer(self):
        samples = list(self._leapquant_iterations)
        return {
            "columns": ["unix_s", "requests", "tokens", "single_token_requests"],
            "iterations": samples,
            "total_observed": self._leapquant_observed,
            "dropped_samples": self._leapquant_observed - len(samples),
        }

    def leapquant_stop_batch_observer(self):
        original = getattr(self, "_leapquant_original_execute", None)
        if original is None:
            raise RuntimeError("LeapQuant batch observer is not installed")
        self.model_runner.execute_model = original
        self._leapquant_original_execute = None
        return {"stopped": True}

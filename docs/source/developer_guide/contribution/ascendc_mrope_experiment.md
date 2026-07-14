# Experimental AscendC MRoPE kernel

This experiment adds an AscendC implementation of the Qwen3 attention preprocessing path:

```text
positions/cos_sin_cache gather + split QKV + Q/K RMSNorm + MRoPE
```

The implementation is intentionally opt-in and keeps the existing Triton kernel as a fallback.
The first version supports the Qwen3-ASR BF16 path: no Q/K bias, no gate, a contiguous
`positions` tensor with shape `[3, num_tokens]`, and head/rope dimensions that are multiples
of 16 and no larger than 256.

## Build

Build on a Linux Ascend development environment with CANN and torch-npu installed:

```bash
cd /vllm-workspace/vllm-ascend
git submodule update --init --recursive
export COMPILE_CUSTOM_KERNELS=1
# Set this explicitly when npu-smi is unavailable. Example for Atlas A2:
export SOC_VERSION=ascend910b1
pip install --no-build-isolation -v -e .
```

Verify registration before starting a model:

```bash
python -c "from vllm_ascend.utils import enable_custom_op; import torch; \
enable_custom_op(); print(hasattr(torch.ops._C_ascend, 'npu_split_qkv_rmsnorm_mrope'))"
```

The command must print `True`.

## Operator correctness and latency

Run the focused E2E test:

```bash
pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_split_qkv_rmsnorm_mrope_ascendc.py
```

Run the standalone probe for decode and prefill shapes:

```bash
python tools/ascendc_mrope_probe.py --num-tokens 1
python tools/ascendc_mrope_probe.py --num-tokens 128
python tools/ascendc_mrope_probe.py --num-tokens 1024
```

The probe compares outputs with Triton and reports latency for Triton (including the external
cache gather) and AscendC (with gather inside the kernel).

## Enable in Qwen3-ASR

The default remains Triton. Enable the experiment before starting vLLM:

```bash
export VLLM_ASCEND_ENABLE_ASCENDC_MROPE=1
vllm serve <Qwen3-ASR-model-path> --dtype bfloat16
```

On the first actual call, the server log must contain:

```text
Using experimental AscendC split QKV + Q/K RMSNorm + MRoPE kernel.
```

If an input is unsupported or the extension was not rebuilt, the log contains:

```text
Experimental AscendC MRoPE kernel is unavailable or the inputs are unsupported; falling back to Triton.
```

Disable the experiment and restore the original behavior with:

```bash
unset VLLM_ASCEND_ENABLE_ASCENDC_MROPE
```

For profiling, collect an msprof or torch-npu profiler trace and search for
`npu_split_qkv_rmsnorm_mrope` or `split_qkv_rmsnorm_mrope_kernel`. Confirm that
`triton_split_qkv_rmsnorm_mrope` is absent from the same Qwen3 attention path when the AscendC
branch is active.

## End-to-end comparison

Use identical model, audio, scheduler, tensor-parallel size, warmup, and request count for both runs:

```bash
# Baseline
unset VLLM_ASCEND_ENABLE_ASCENDC_MROPE
# Start the server and record transcription, TTFT, ITL, throughput, and peak memory.

# Experiment
export VLLM_ASCEND_ENABLE_ASCENDC_MROPE=1
# Restart the server and repeat the same requests.
```

Do not use latency from the first request as the final result. Warm up both implementations and
compare multiple decode token counts and audio lengths. The change is only useful if transcription
accuracy remains stable and end-to-end performance improves, not merely if a microbenchmark is faster.

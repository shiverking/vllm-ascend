"""Correctness and latency probe for the experimental AscendC MRoPE op."""

import argparse
import time
from collections.abc import Callable

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op, is_310p


def timed_ms(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="auto selects float16 on 310P and bfloat16 on other Ascend devices",
    )
    args = parser.parse_args()

    dtype_name = "float16" if args.dtype == "auto" and is_310p() else args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16"
    if is_310p() and dtype_name == "bfloat16":
        raise ValueError("Ascend 310P supports only float16 in this experimental kernel")
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16

    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "npu_split_qkv_rmsnorm_mrope"):
        raise RuntimeError(
            "AscendC MRoPE op is not registered. Rebuild vllm-ascend with "
            "COMPILE_CUSTOM_KERNELS=1 before running this probe."
        )

    torch.set_default_device("npu")
    init_device_properties_triton()
    torch.manual_seed(0)

    mrope_section = [16, 24, 24]
    rope_dim = 2 * sum(mrope_section)
    q_size = args.num_q_heads * args.head_size
    kv_size = args.num_kv_heads * args.head_size
    qkv = torch.randn(
        args.num_tokens,
        q_size + 2 * kv_size,
        dtype=dtype,
        device="npu",
    )
    q_weight = torch.randn(args.head_size, dtype=dtype, device="npu")
    k_weight = torch.randn(args.head_size, dtype=dtype, device="npu")
    cos_sin_cache = torch.randn(
        args.max_positions,
        rope_dim,
        dtype=dtype,
        device="npu",
    )
    positions = torch.randint(
        0,
        args.max_positions,
        (3, args.num_tokens),
        dtype=torch.int64,
        device="npu",
    )

    def run_triton():
        cos_sin = cos_sin_cache[positions]
        return torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin=cos_sin,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            eps=1e-6,
            mrope_section=mrope_section,
            is_interleaved=args.interleaved,
            rope_dim=rope_dim,
        )[:3]

    def run_ascendc():
        return torch.ops._C_ascend.npu_split_qkv_rmsnorm_mrope(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_size,
            1e-6,
            mrope_section,
            args.interleaved,
            rope_dim,
        )

    triton_outputs = run_triton()
    ascendc_outputs = run_ascendc()
    for name, ascendc_output, triton_output in zip(
        ("q", "k", "v"), ascendc_outputs, triton_outputs
    ):
        torch.testing.assert_close(
            ascendc_output,
            triton_output,
            atol=0 if name == "v" else (3e-2 if dtype == torch.float16 else 2e-2),
            rtol=0 if name == "v" else (3e-2 if dtype == torch.float16 else 2e-2),
        )

    triton_ms = timed_ms(run_triton, args.warmup, args.iterations)
    ascendc_ms = timed_ms(run_ascendc, args.warmup, args.iterations)
    print(f"AscendC operator is registered and produced matching {dtype_name} outputs.")
    print(f"Triton (including cache gather): {triton_ms:.4f} ms")
    print(f"AscendC (gather fused):          {ascendc_ms:.4f} ms")
    print(f"Speedup:                         {triton_ms / ascendc_ms:.3f}x")


if __name__ == "__main__":
    main()

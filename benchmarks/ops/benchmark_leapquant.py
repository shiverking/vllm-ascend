# SPDX-License-Identifier: Apache-2.0
"""Synthetic full-window GDN timing, independent of application data or code."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.ops.leapquant_native import SERVING_PACKET_BYTES, NativeLeapQuant


def inputs(batch):
    return (
        torch.nn.functional.normalize(torch.randn(batch, 16, 128, device="npu"), dim=-1).bfloat16(),
        torch.nn.functional.normalize(torch.randn(batch, 16, 128, device="npu"), dim=-1).bfloat16(),
        torch.randn(batch, 32, 128, device="npu").bfloat16(),
        torch.rand(batch, 32, device="npu").bfloat16(),
        -torch.rand(batch, 32, device="npu") * 0.2,
    )


def measure(cycle, samples, replays):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cycle()
    for _ in range(3):
        graph.replay()
    elapsed = []
    for _ in range(samples):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) / (replays * 16))
    return {"median_ms_per_step": statistics.median(elapsed), "samples_ms_per_step": elapsed}


def run_mode(initial, indices, tokens, backend, samples, replays):
    if backend is None:
        state = initial.clone()
        lengths = torch.arange(initial.shape[0] + 1, dtype=torch.int32, device="npu")

        def step(args):
            q, k, v, beta, g = args
            return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=q,
                key=k,
                value=v,
                beta=beta,
                g=g,
                state=state,
                scale=128**-0.5,
                actual_seq_lengths=lengths,
                ssm_state_indices=indices,
            )
    else:
        state = torch.zeros(initial.shape[0], 32, SERVING_PACKET_BYTES, dtype=torch.uint8, device="npu")
        backend.encode_state(initial, state, indices)

        def step(args):
            return backend.decode(*args, state, indices)

    def cycle():
        for args in tokens:
            step(args)

    for _ in range(2):
        cycle()
    return measure(cycle, samples, replays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", action="append", required=True, type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 192, 256])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--cores", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.batches, args.samples, args.replays) < 1:
        parser.error("batch sizes, samples and replays must be positive")
    torch.npu.set_device(0)
    torch.manual_seed(20261108)
    libraries = [p.resolve(strict=True) for p in args.library]
    backends = {f"native_{i}": NativeLeapQuant(str(p), cores=args.cores) for i, p in enumerate(libraries)}
    result = {
        "protocol": {
            "window": 16,
            "distinct_inputs_per_window": 16,
            "includes_real_fold": True,
            "serving_packet_bytes": SERVING_PACKET_BYTES,
            "cores": args.cores,
            "scope": "synthetic operator timing, not model throughput or accuracy validation",
        },
        "libraries_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries},
        "benchmarks": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        initial = torch.randn(batch, 32, 128, 128, device="npu") * 0.01
        indices = torch.arange(batch, dtype=torch.int32, device="npu")
        tokens = [inputs(batch) for _ in range(16)]
        for name, backend in [("fp32_gdn", None), *backends.items()]:
            row = {
                "batch": batch,
                "mode": name,
                **run_mode(initial, indices, tokens, backend, args.samples, args.replays),
            }
            result["benchmarks"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

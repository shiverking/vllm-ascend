from collections.abc import Callable
from statistics import median

import pytest
import torch
import torch_npu

from vllm_ascend._310p.audio_encoder_acl_graph import (
    build_padded_attention_mask,
)
from vllm_ascend.utils import is_310p as is_310p_hw


torch_npu.npu.set_compile_mode(jit_compile=False)

GRAPH_SIZE = 128
NUM_HEADS = 20
HEAD_SIZE = 64
SCALE = HEAD_SIZE**-0.5
TOKEN_SIZES = (26, 52, 78, 104)
WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 100


def _benchmark_npu(fn: Callable[[], object]) -> tuple[float, float, float]:
    for _ in range(WARMUP_ITERATIONS):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    elapsed_ms = []
    for _ in range(BENCHMARK_ITERATIONS):
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    elapsed_ms.sort()
    p90_index = int(0.9 * (len(elapsed_ms) - 1))
    return median(elapsed_ms), elapsed_ms[0], elapsed_ms[p90_index]


def _prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return torch_npu.npu_prompt_flash_attention(
        query,
        key,
        value,
        atten_mask=mask,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        scale_value=SCALE,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BNSD",
        sparse_mode=0,
    )


def _unpad_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_lengths: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    torch_npu._npu_flash_attention_unpad(
        query=query,
        key=key,
        value=value,
        seq_len=sequence_lengths,
        scale_value=SCALE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        out=output,
    )
    return output


def _print_results(rows: list[dict[str, float]]) -> None:
    print("\nAudio Encoder Attention operator benchmark on Ascend 310P")
    print(
        "Times are Device event milliseconds per call; mask construction, "
        "mask H2D, QKV layout conversion, and graph capture are excluded."
    )
    print(
        f"warmup={WARMUP_ITERATIONS}, iterations={BENCHMARK_ITERATIONS}, "
        f"heads={NUM_HEADS}, head_size={HEAD_SIZE}, graph_size={GRAPH_SIZE}"
    )
    print(
        "tokens | unpad eager p50/min/p90 | PFA eager p50/min/p90 | "
        "PFA graph p50/min/p90 | eager speedup | graph speedup"
    )
    for row in rows:
        print(
            f"{int(row['tokens']):>6} | "
            f"{row['unpad_p50']:.4f}/{row['unpad_min']:.4f}/"
            f"{row['unpad_p90']:.4f} | "
            f"{row['pfa_p50']:.4f}/{row['pfa_min']:.4f}/"
            f"{row['pfa_p90']:.4f} | "
            f"{row['graph_p50']:.4f}/{row['graph_min']:.4f}/"
            f"{row['graph_p90']:.4f} | "
            f"{row['unpad_p50'] / row['pfa_p50']:.3f}x | "
            f"{row['unpad_p50'] / row['graph_p50']:.3f}x"
        )


@pytest.mark.skipif(
    not is_310p_hw(),
    reason="Tested separately on an Ascend 310P machine.",
)
@torch.inference_mode()
def test_audio_encoder_attention_operator_performance_310p():
    torch.manual_seed(0)
    query_tnd = torch.randn(
        GRAPH_SIZE,
        NUM_HEADS,
        HEAD_SIZE,
        device="npu",
        dtype=torch.float16,
    )
    key_tnd = torch.randn_like(query_tnd)
    value_tnd = torch.randn_like(query_tnd)
    query_bnsd = query_tnd.transpose(0, 1).unsqueeze(0).contiguous()
    key_bnsd = key_tnd.transpose(0, 1).unsqueeze(0).contiguous()
    value_bnsd = value_tnd.transpose(0, 1).unsqueeze(0).contiguous()

    masks = {
        tokens: build_padded_attention_mask((tokens,))[0].npu()
        for tokens in TOKEN_SIZES
    }
    static_query = query_bnsd.clone()
    static_key = key_bnsd.clone()
    static_value = value_bnsd.clone()
    static_mask = build_padded_attention_mask((GRAPH_SIZE,))[0].npu()

    for _ in range(2):
        _prompt_attention(
            static_query,
            static_key,
            static_value,
            static_mask,
        )
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = _prompt_attention(
            static_query,
            static_key,
            static_value,
            static_mask,
        )

    rows = []
    for tokens in TOKEN_SIZES:
        sequence_lengths = torch.tensor([tokens], dtype=torch.int32)
        query = query_tnd[:tokens].contiguous()
        key = key_tnd[:tokens].contiguous()
        value = value_tnd[:tokens].contiguous()
        unpad_output = torch.empty_like(query)
        mask = masks[tokens]

        eager_unpad = _unpad_attention(
            query,
            key,
            value,
            sequence_lengths,
            unpad_output,
        ).clone()
        eager_pfa = _prompt_attention(
            query_bnsd,
            key_bnsd,
            value_bnsd,
            mask,
        )
        static_mask.copy_(mask)
        graph.replay()
        torch.npu.synchronize()
        replay_pfa = graph_output.clone()

        eager_pfa_prefix = eager_pfa[0, :, :tokens].transpose(0, 1)
        replay_pfa_prefix = replay_pfa[0, :, :tokens].transpose(0, 1)
        torch.testing.assert_close(
            eager_pfa_prefix,
            eager_unpad,
            atol=3e-2,
            rtol=3e-2,
        )
        torch.testing.assert_close(
            replay_pfa_prefix,
            eager_unpad,
            atol=3e-2,
            rtol=3e-2,
        )

        unpad_p50, unpad_min, unpad_p90 = _benchmark_npu(
            lambda: _unpad_attention(
                query,
                key,
                value,
                sequence_lengths,
                unpad_output,
            )
        )
        pfa_p50, pfa_min, pfa_p90 = _benchmark_npu(
            lambda: _prompt_attention(
                query_bnsd,
                key_bnsd,
                value_bnsd,
                mask,
            )
        )
        static_mask.copy_(mask)
        torch.npu.synchronize()
        graph_p50, graph_min, graph_p90 = _benchmark_npu(graph.replay)
        rows.append(
            {
                "tokens": float(tokens),
                "unpad_p50": unpad_p50,
                "unpad_min": unpad_min,
                "unpad_p90": unpad_p90,
                "pfa_p50": pfa_p50,
                "pfa_min": pfa_min,
                "pfa_p90": pfa_p90,
                "graph_p50": graph_p50,
                "graph_min": graph_min,
                "graph_p90": graph_p90,
            }
        )

    _print_results(rows)

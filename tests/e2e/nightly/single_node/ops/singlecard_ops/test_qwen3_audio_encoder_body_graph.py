"""910B integration gate for the Qwen3-ASR audio encoder body graph pool.

This uses a small encoder body so the graph planner, masked attention, stream
ordering, output split, and static-buffer lifetime can be tested together.
Real-weight REST and WebSocket requests are still required after this gate.
"""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention  # noqa: E402
from vllm_ascend.utils import is_310p  # noqa: E402
from vllm_ascend.worker.qwen3_audio_encoder_graph import (  # noqa: E402
    Qwen3AudioEncoderGraphPool,
    plan_audio_graph_chunks,
)

pytestmark = pytest.mark.skipif(is_310p(), reason="910B audio body graph gate")

WINDOW = 104
BUDGETS = (26, 104, 208, 416)
HIDDEN_SIZE = 256
TOKEN_COUNTS = (25, 26, 27, 52, 78, 104, 105, 128, 208, 256, 384, 416, 512)


class _AudioTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_post = torch.nn.LayerNorm(HIDDEN_SIZE)
        self.proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.attn = AscendMMEncoderAttention(num_heads=4, head_size=64)

    def get_attention_window_tokens(self) -> int:
        return WINDOW

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> None:
        return None

    def _forward_encoder_body(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: None,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        qkv = hidden_states.unsqueeze(0)
        attended = self.attn.forward_oot(
            qkv, qkv, qkv, cu_seqlens, max_seqlen, sequence_lengths
        ).squeeze(0)
        return self.proj(self.ln_post(hidden_states + attended))


class _Model:
    def __init__(self):
        self.audio_tower = _AudioTower().to(device="npu", dtype=torch.float16)

    def _prepare_audio_encoder_graph_inputs(self, mm_kwargs):
        topology = mm_kwargs["topology"]
        cu_seqlens = torch.tensor(
            [0, *accumulate(topology)], device="npu", dtype=torch.int32
        )
        sequence_lengths = torch.tensor(topology, device="cpu", dtype=torch.int32)
        return mm_kwargs["hidden_states"], cu_seqlens, None, sequence_lengths

    def get_encoder_cudagraph_item_specs(self, mm_kwargs):
        return [
            SimpleNamespace(output_tokens=length)
            for length in mm_kwargs["item_lengths"]
        ]


def _topology(tokens: int) -> tuple[int, ...]:
    full, tail = divmod(tokens, WINDOW)
    return (WINDOW,) * full + ((tail,) if tail else ())


@torch.inference_mode()
def test_qwen3_audio_body_graph_pool_matches_eager_across_shapes():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")

    torch.manual_seed(0)
    model = _Model()
    pool = Qwen3AudioEncoderGraphPool(model, torch.device("npu"), BUDGETS)
    pool.capture()
    torch.npu.synchronize()

    first_output = None
    first_snapshot = None
    for tokens in (*TOKEN_COUNTS, 25):
        topology = _topology(tokens)
        hidden_states = torch.randn(
            tokens, HIDDEN_SIZE, device="npu", dtype=torch.float16
        )
        item_lengths = (tokens,) if tokens != 128 else (104, 24)
        mm_kwargs = {
            "hidden_states": hidden_states,
            "topology": topology,
            "item_lengths": item_lengths,
        }
        plan = plan_audio_graph_chunks(topology, BUDGETS)
        assert all(chunk.budget is not None for chunk in plan)

        _, cu_seqlens, _, sequence_lengths = (
            model._prepare_audio_encoder_graph_inputs(mm_kwargs)
        )
        expected = model.audio_tower._forward_encoder_body(
            hidden_states, cu_seqlens, None, sequence_lengths
        )
        actual_parts = pool.execute(mm_kwargs)
        actual = torch.cat(actual_parts, dim=0)
        torch.npu.synchronize()

        assert [part.shape[0] for part in actual_parts] == list(item_lengths)
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        if first_output is None:
            first_output = actual
            first_snapshot = actual.clone()
        else:
            torch.testing.assert_close(first_output, first_snapshot, atol=0, rtol=0)
        print(
            f"[AUDIO_BODY_GRAPH_GATE] passed tokens={tokens} "
            f"topology={topology} graphs={[chunk.budget for chunk in plan]}",
            flush=True,
        )

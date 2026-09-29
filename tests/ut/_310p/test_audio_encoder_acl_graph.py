from unittest.mock import MagicMock

import torch

from vllm_ascend._310p.audio_encoder_acl_graph import (
    AudioEncoderAclGraphPool,
    build_padded_attention_mask,
)


def test_audio_graph_plan_keeps_attention_sequences_whole():
    pool = AudioEncoderAclGraphPool(
        encoder=MagicMock(),
        eager_forward=MagicMock(),
        graph_sizes=(26, 104, 208, 416),
    )
    plan = pool._build_plan((104, 24))
    assert [chunk.runner.num_tokens for chunk in plan] == [104, 26]
    assert [(chunk.sequence_start, chunk.sequence_end) for chunk in plan] == [
        (0, 1),
        (1, 2),
    ]


def test_audio_graph_mask_isolates_padding():
    mask, padding = build_padded_attention_mask((104, 24), 208)
    assert padding == 80
    assert mask.shape == (208, 208)
    assert mask[:104, 104:].all()
    assert mask[104:128, 128:].all()
    assert not mask[128:, 128:].any()
    assert mask.dtype == torch.bool

# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.worker.fixed_prefix import copy_fixed_prefix_pages


def test_copy_partial_page_and_recurrent_state_are_private():
    # Two kernel blocks per attention scheduler page, one row per Mamba page.
    attention = torch.arange(48, dtype=torch.float32).reshape(12, 2, 1, 2)
    recurrent = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    groups = [
        SimpleNamespace(
            layer_names=["attn"], kv_cache_spec=FullAttentionSpec(4, num_kv_heads=1, head_size=2, dtype=torch.float32)
        ),
        SimpleNamespace(
            layer_names=["gdn"],
            kv_cache_spec=MambaSpec(4, shapes=((3,),), dtypes=(torch.float32,), mamba_cache_mode="align"),
        ),
    ]
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attn": SimpleNamespace(kv_cache=(attention, attention.clone())),
                "gdn": SimpleNamespace(kv_cache=[recurrent]),
            }
        ),
    )
    original_attention = attention[2:4].clone()
    original_state = recurrent[1].clone()
    copy_fixed_prefix_pages(runner, [(0, 1, 4), (1, 1, 4)])
    torch.testing.assert_close(attention[8:10], original_attention)
    torch.testing.assert_close(recurrent[4], original_state)
    attention[8:10].zero_()
    recurrent[4].zero_()
    torch.testing.assert_close(attention[2:4], original_attention)
    torch.testing.assert_close(recurrent[1], original_state)

from types import SimpleNamespace
from unittest.mock import MagicMock

from torch import nn

from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)


def test_mtp_shares_target_lm_head_for_dense_model():
    target_head = nn.Linear(4, 8, bias=False)
    draft_head = nn.Linear(4, 8, bias=False)
    proposer = SimpleNamespace(
        method="mtp",
        model=SimpleNamespace(
            lm_head=draft_head,
            model=SimpleNamespace(layers=nn.ModuleList([nn.Identity()])),
        ),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=MagicMock(has_full_cudagraphs=lambda: False)
            )
        ),
        use_cuda_graph=False,
    )
    target = SimpleNamespace(lm_head=target_head)

    AscendSpecDecodeBaseProposer._maybe_share_lm_head(proposer, target)

    assert proposer.model.lm_head is target_head


def test_mtp_replaces_per_layer_shared_head_when_present():
    target_head = nn.Linear(4, 8, bias=False)
    shared_head = SimpleNamespace(head=nn.Linear(4, 8, bias=False))
    proposer = SimpleNamespace(
        method="mtp",
        model=SimpleNamespace(
            lm_head=nn.Linear(4, 8, bias=False),
            model=SimpleNamespace(
                layers=nn.ModuleDict(
                    {"0": nn.Module()}
                )
            ),
        ),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=MagicMock(has_full_cudagraphs=lambda: False)
            )
        ),
        use_cuda_graph=False,
    )
    proposer.model.model.layers["0"].shared_head = shared_head

    AscendSpecDecodeBaseProposer._maybe_share_lm_head(
        proposer, SimpleNamespace(lm_head=target_head)
    )

    assert shared_head.head is target_head

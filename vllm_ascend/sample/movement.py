# SPDX-License-Identifier: Apache-2.0
"""CPU metadata eligibility; no device reads or mutation."""


def inactive_builtin(processor):
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinPLogitsProcessor,
        MinTokensLogitsProcessor,
    )

    if type(processor) is MinTokensLogitsProcessor:
        return not processor.min_toks
    if type(processor) is LogitBiasLogitsProcessor:
        return not processor.biases
    if type(processor) is MinPLogitsProcessor:
        return processor.min_p_count == 0
    return False


def can_sample_without_cast(metadata, predict_bonus_token=False):
    holder = metadata.thinking_budget_state_holder
    return (
        not predict_bonus_token
        and metadata.all_greedy
        and metadata.no_penalties
        and metadata.max_num_logprobs is None
        and not metadata.logprob_token_ids
        and metadata.allowed_token_ids_mask is None
        and not metadata.bad_words_token_ids
        and all(inactive_builtin(p) for p in metadata.logitsprocs.non_argmax_invariant)
        and all(inactive_builtin(p) for p in metadata.logitsprocs.argmax_invariant)
        and not (holder is not None and holder.has_tracked_requests())
    )

# SPDX-License-Identifier: Apache-2.0
"""Copy immutable prefix tails into private continuation pages on device."""

from vllm.v1.kv_cache_interface import MambaSpec


def copy_fixed_prefix_pages(runner, copies):
    """Run on the model stream before recurrent-state preprocessing.

    Attention may expose several kernel blocks per scheduler page. Recurrent
    states always use one tensor row per scheduler page. No host staging occurs.
    """
    for group_id, source, target in copies:
        group = runner.kv_cache_config.kv_cache_groups[group_id]
        for name in group.layer_names:
            tensors = runner.compilation_config.static_forward_context[name].kv_cache
            if not isinstance(tensors, (list, tuple)):
                raise ValueError("Fixed prefix requires explicit K/V or state tensors")
            for tensor in tensors:
                if isinstance(group.kv_cache_spec, MambaSpec):
                    tensor[target].copy_(tensor[source])
                else:
                    # Ascend K/V views: [kernel_blocks, block_size, heads, dim].
                    kernel_size = tensor.shape[1]
                    if group.kv_cache_spec.block_size % kernel_size:
                        raise ValueError("Unsupported attention page layout")
                    ratio = group.kv_cache_spec.block_size // kernel_size
                    tensor[target * ratio : (target + 1) * ratio].copy_(tensor[source * ratio : (source + 1) * ratio])

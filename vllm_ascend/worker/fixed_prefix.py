# SPDX-License-Identifier: Apache-2.0
"""Copy immutable prefix tails into private continuation pages on device."""

from bisect import bisect_left

import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.ops.data_movement_config import DataMovementConfig
from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel


def independent_copy_ranges(descriptors):
    """Parallel copies require disjoint writes and immutable source ranges."""
    reads = []
    for source, _, size in sorted(descriptors):
        if reads and source <= reads[-1][1]:
            reads[-1] = (reads[-1][0], max(reads[-1][1], source + size))
        else:
            reads.append((source, source + size))
    starts = [start for start, _ in reads]
    previous_end = -1
    for target, size in sorted((dst, size) for _, dst, size in descriptors):
        if target < previous_end:
            return False
        previous_end = target + size
        index = bisect_left(starts, previous_end) - 1
        if index >= 0 and reads[index][1] > target:
            return False
    return True


def copy_fixed_prefix_pages(runner, copies):
    """Run on the model stream before recurrent-state preprocessing.

    Attention may expose several kernel blocks per scheduler page. Recurrent
    states always use one tensor row per scheduler page. No host staging occurs.
    """
    flags = (
        DataMovementConfig.from_config(runner.vllm_config) if hasattr(runner, "vllm_config") else DataMovementConfig()
    )
    pairs = []
    for group_id, source, target in copies:
        group = runner.kv_cache_config.kv_cache_groups[group_id]
        for name in group.layer_names:
            tensors = runner.compilation_config.static_forward_context[name].kv_cache
            if not isinstance(tensors, (list, tuple)):
                raise ValueError("Fixed prefix requires explicit K/V or state tensors")
            for tensor in tensors:
                if isinstance(group.kv_cache_spec, MambaSpec):
                    pairs.append((tensor[source], tensor[target]))
                else:
                    # Ascend K/V views: [kernel_blocks, block_size, heads, dim].
                    kernel_size = tensor.shape[1]
                    if group.kv_cache_spec.block_size % kernel_size:
                        raise ValueError("Unsupported attention page layout")
                    ratio = group.kv_cache_spec.block_size // kernel_size
                    pairs.append(
                        (tensor[source * ratio : (source + 1) * ratio], tensor[target * ratio : (target + 1) * ratio])
                    )

    if not flags.batched_prefix_copy:
        for source, target in pairs:
            target.copy_(source)
        return
    # Exact descriptor deduplication only. Different views of shared storage
    # must not be deduplicated based on the storage base pointer alone.
    descriptors = list(
        dict.fromkeys(
            (src.data_ptr(), dst.data_ptr(), src.numel() * src.element_size())
            for src, dst in pairs
            if src.data_ptr() != dst.data_ptr()
        )
    )
    if not descriptors:
        return
    if not independent_copy_ranges(descriptors) or any(
        not src.is_contiguous() or not dst.is_contiguous() for src, dst in pairs
    ):
        for source, target in pairs:
            target.copy_(source)
        return
    import torch_npu

    if any(
        torch_npu.get_npu_format(src) not in (0, 2) or torch_npu.get_npu_format(dst) not in (0, 2) for src, dst in pairs
    ):
        for source, target in pairs:
            target.copy_(source)
        return

    srcs, dsts, sizes = zip(*descriptors)
    cpu = torch.tensor([srcs, dsts, sizes], dtype=torch.int64, pin_memory=True)
    device = cpu.to(pairs[0][0].device, non_blocking=True)
    batch_memcpy_kernel[(len(descriptors),)](device[0], device[1], device[2], BLOCK_SIZE=8192)

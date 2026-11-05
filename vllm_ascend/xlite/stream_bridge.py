# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Small version-pinned stream binding for native xLite standalone operators."""

from importlib import metadata
from pathlib import Path


def load_stream_bridge():
    # Compilation is worker/startup only, never in a model forward.
    from torch.utils.cpp_extension import load

    version = metadata.version("xlite")
    if version != "0.2.0rc4":
        raise RuntimeError(f"xLite stream bridge validated only for 0.2.0rc4, got {version}")
    distribution = metadata.distribution("xlite")
    headers = distribution.locate_file("xlite/include")
    return load(
        name="vllm_ascend_xlite_stream_rc4",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        extra_include_paths=[str(headers)],
        extra_cflags=["-O2"],
        extra_ldflags=[
            "-L" + str(distribution.locate_file("xlite/lib")),
            "-lxlite",
            "-Wl,-rpath," + str(distribution.locate_file("xlite/lib")),
        ],
        with_cuda=False,
        verbose=False,
    )

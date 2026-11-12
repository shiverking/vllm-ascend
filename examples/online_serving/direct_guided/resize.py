# SPDX-License-Identifier: Apache-2.0
"""Small dependency-free counterpart of qwen_vl_utils.smart_resize.

Used only when qwen_vl_utils isn't installed. Preserves factor rounding,
minimum/maximum pixel budget and the aspect-ratio bound.
"""

from __future__ import annotations

import math


def smart_resize(
    height: int, width: int, *, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
) -> tuple[int, int]:
    if height <= 0 or width <= 0 or factor <= 0:
        raise ValueError("height, width and factor must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio must not exceed 200")
    if min_pixels < factor * factor or max_pixels < min_pixels:
        raise ValueError("invalid pixel budget")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = max(factor, math.ceil(height * beta / factor) * factor)
        w = max(factor, math.ceil(width * beta / factor) * factor)
    return int(h), int(w)

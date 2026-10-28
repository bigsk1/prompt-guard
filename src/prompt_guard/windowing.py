"""Overlapping token windows with provable coverage.

`window_size` is the total sequence length, including the tokenizer's special
tokens. `overlap` is the number of content tokens shared by consecutive
windows, matching the tokenizer `stride` used by the model card.
"""

import math
from collections.abc import Sequence


def window_count(token_count: int, capacity: int, step: int) -> int:
    """How many content windows cover `token_count` tokens."""
    if capacity < 1 or step < 1:
        raise ValueError("capacity and step must be positive")
    if token_count <= 0:
        return 1
    if token_count <= capacity:
        return 1
    return math.ceil((token_count - capacity) / step) + 1


def plan_windows(
    content_ids: Sequence[int],
    *,
    capacity: int,
    overlap: int,
    prefix: Sequence[int],
    suffix: Sequence[int],
) -> list[list[int]]:
    """Wrap every content token in at least one special-token window."""
    if capacity < 1:
        raise ValueError("content capacity must be positive")
    if overlap < 0 or overlap >= capacity:
        raise ValueError("overlap must be >= 0 and smaller than the content capacity")
    step = capacity - overlap
    prefix_ids = list(prefix)
    suffix_ids = list(suffix)
    tokens = list(content_ids)
    if not tokens:
        return [prefix_ids + suffix_ids]

    windows: list[list[int]] = []
    start = 0
    total = len(tokens)
    while True:
        end = min(start + capacity, total)
        windows.append(prefix_ids + tokens[start:end] + suffix_ids)
        if end >= total:
            break
        start += step
    return windows


def content_is_covered(
    content_ids: Sequence[int],
    windows: Sequence[Sequence[int]],
    *,
    prefix_len: int,
    suffix_len: int,
    capacity: int,
    step: int,
) -> bool:
    """True when every content token appears in the planned windows."""
    if len(windows) != window_count(len(content_ids), capacity, step):
        return False
    if not content_ids:
        return len(windows) == 1 and len(windows[0]) == prefix_len + suffix_len
    covered = [0] * len(content_ids)
    start = 0
    for window in windows:
        inner = _inner(window, prefix_len, suffix_len)
        end = start + len(inner)
        if end > len(content_ids) or list(content_ids[start:end]) != inner:
            return False
        for offset in range(len(inner)):
            covered[start + offset] += 1
        if end >= len(content_ids):
            return all(count >= 1 for count in covered)
        start += step
    return False


def _inner(window: Sequence[int], prefix_len: int, suffix_len: int) -> list[int]:
    if suffix_len:
        return list(window[prefix_len:-suffix_len])
    return list(window[prefix_len:])

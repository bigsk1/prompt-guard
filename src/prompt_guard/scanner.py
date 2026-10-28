"""Tokenize with the bundled normalizer, window, and score.

The scan function receives text and a deadline callback. Caller metadata such
as `source` and `threshold` never reaches this module.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from prompt_guard.errors import ScanError, input_too_long, internal_error, scan_timeout, too_many_windows
from prompt_guard.scoring import injection_score
from prompt_guard.windowing import plan_windows, window_count


class LogitsModel:
    """One loaded classifier. Implementations must not retain text."""

    def predict_logits(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        should_stop: Callable[[], bool],
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class ScanSuccess:
    injection_score: float
    windows_scanned: int
    windows_total: int
    complete: bool
    truncated: bool


@dataclass(frozen=True)
class WindowPlan:
    capacity: int
    overlap: int
    step: int
    prefix: tuple[int, ...]
    suffix: tuple[int, ...]
    batch_size: int
    max_windows: int
    max_input_tokens: int
    pad_id: int
    injection_index: int


class Scanner:
    def __init__(self, tokenizer: object, model: LogitsModel, plan: WindowPlan) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._plan = plan

    def scan(self, text: str, should_stop: Callable[[], bool]) -> ScanSuccess:
        if should_stop():
            raise scan_timeout()
        content = content_token_ids(self._tokenizer, text)
        plan = self._plan
        if len(content) > plan.max_input_tokens:
            raise input_too_long()
        total = window_count(len(content), plan.capacity, plan.step)
        if total > plan.max_windows:
            raise too_many_windows()
        windows = plan_windows(
            content,
            capacity=plan.capacity,
            overlap=plan.overlap,
            prefix=plan.prefix,
            suffix=plan.suffix,
        )
        if len(windows) != total:
            raise internal_error()
        if should_stop():
            raise scan_timeout()

        scores: list[float] = []
        scanned = 0
        for batch in _batches(windows, plan.batch_size):
            if should_stop():
                raise scan_timeout()
            input_ids, attention_mask = collate(batch, plan.pad_id)
            logits = self._model.predict_logits(input_ids, attention_mask, should_stop)
            if should_stop():
                raise scan_timeout()
            try:
                scores.append(injection_score(logits, plan.injection_index))
            except ValueError as exc:
                raise internal_error() from exc
            scanned += len(batch)
        if scanned != total or not scores:
            raise internal_error()
        return ScanSuccess(
            injection_score=max(scores),
            windows_scanned=scanned,
            windows_total=total,
            complete=True,
            truncated=False,
        )


def content_token_ids(tokenizer: object, text: str) -> list[int]:
    """Token IDs after the tokenizer's own normalizer, without special tokens.

    The backend encode path avoids the transformers length warning. That warning
    claims a long sequence will be fed to the model; this service windows it
    instead and must not silently truncate.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        # The published tokenizer enables truncation at 1024. Leaving it on
        # would silently drop the rest of a long document and look like coverage.
        if getattr(backend, "truncation", None):
            backend.no_truncation()
        return list(backend.encode(text, add_special_tokens=False).ids)
    encoded = tokenizer(  # type: ignore[operator]
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def discover_special_tokens(tokenizer: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Learn the special-token wrapper from the tokenizer, not from constants."""
    probe = "coverage-probe"
    full = _call_ids(tokenizer, probe, add_special_tokens=True)
    content = content_token_ids(tokenizer, probe)
    if not content:
        raise ScanError(
            "internal_error",
            "The tokenizer produced no tokens for the special-token probe.",
            500,
            False,
        )
    found: int | None = None
    for start in range(len(full) - len(content) + 1):
        if full[start : start + len(content)] == content:
            if found is not None:
                raise ScanError(
                    "internal_error",
                    "The tokenizer special-token wrapper is ambiguous.",
                    500,
                    False,
                )
            found = start
    if found is None:
        raise ScanError(
            "internal_error",
            "The tokenizer special-token wrapper could not be determined.",
            500,
            False,
        )
    prefix = tuple(full[:found])
    suffix = tuple(full[found + len(content) :])
    declared = int(tokenizer.num_special_tokens_to_add(False))  # type: ignore[attr-defined]
    if declared != len(prefix) + len(suffix):
        raise ScanError(
            "internal_error",
            "The tokenizer special-token count does not match its encodings.",
            500,
            False,
        )
    return prefix, suffix


def collate(windows: Sequence[Sequence[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(window) for window in windows)
    input_ids = np.full((len(windows), width), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(windows), width), dtype=np.int64)
    for row, window in enumerate(windows):
        input_ids[row, : len(window)] = window
        attention_mask[row, : len(window)] = 1
    return input_ids, attention_mask


def _call_ids(tokenizer: object, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(  # type: ignore[operator]
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _batches(windows: Sequence[list[int]], batch_size: int) -> list[list[list[int]]]:
    return [list(windows[start : start + batch_size]) for start in range(0, len(windows), batch_size)]

"""Tokenize with the bundled normalizer, window, and score.

The scan function receives text and a deadline callback. Caller metadata such
as `source` and `threshold` never reaches this module.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from prompt_guard.errors import ScanError, input_too_long, internal_error, scan_timeout, too_many_windows
from prompt_guard.scoring import injection_scores
from prompt_guard.windowing import content_token_spans, plan_windows, window_count


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
class WindowObservation:
    index: int
    content_token_start: int
    content_token_end: int
    char_start: int | None
    char_end: int | None
    injection_score: float


@dataclass(frozen=True)
class ScanSuccess:
    injection_score: float
    windows_scanned: int
    windows_total: int
    complete: bool
    truncated: bool
    windows: tuple[WindowObservation, ...]
    winning_window_index: int
    offset_unit: str | None
    char_offset_reason: str | None


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
        tokenized = tokenize_content(self._tokenizer, text)
        content = tokenized.ids
        plan = self._plan
        if len(content) > plan.max_input_tokens:
            raise input_too_long()
        total = window_count(len(content), plan.capacity, plan.step)
        if total > plan.max_windows:
            raise too_many_windows()
        spans = content_token_spans(len(content), capacity=plan.capacity, overlap=plan.overlap)
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
                row_scores = injection_scores(logits, plan.injection_index)
            except ValueError as exc:
                raise internal_error() from exc
            if len(row_scores) != len(batch):
                raise internal_error()
            scores.extend(row_scores)
            scanned += len(batch)
        if scanned != total or len(scores) != total or len(spans) != total:
            raise internal_error()
        observations = tuple(
            _observation(index, span, tokenized, scores[index])
            for index, span in enumerate(spans)
        )
        winning = max(range(total), key=lambda index: scores[index])
        reason = tokenized.char_offset_reason
        if reason is None and any(
            item.char_start is None and item.content_token_end > item.content_token_start
            for item in observations
        ):
            reason = "tokenizer_offset_did_not_cover_window"
        return ScanSuccess(
            injection_score=scores[winning],
            windows_scanned=scanned,
            windows_total=total,
            complete=True,
            truncated=False,
            windows=observations,
            winning_window_index=winning,
            offset_unit=tokenized.offset_unit,
            char_offset_reason=reason,
        )


@dataclass(frozen=True)
class TokenizedText:
    ids: list[int]
    offsets: tuple[tuple[int, int], ...] | None
    offset_unit: str | None
    char_offset_reason: str | None


def tokenize_content(tokenizer: object, text: str) -> TokenizedText:
    """Content tokens plus original-text offsets when the tokenizer provides them."""
    custom = getattr(tokenizer, "encode_content", None)
    if custom is not None:
        encoded = custom(text)
        ids = [int(token_id) for token_id in encoded["input_ids"]]
        return _with_offsets(text, ids, encoded.get("offsets"))
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        # The published tokenizer enables truncation at 1024. Leaving it on
        # would silently drop the rest of a long document and look like coverage.
        if getattr(backend, "truncation", None):
            backend.no_truncation()
        encoding = backend.encode(text, add_special_tokens=False)
        return _with_offsets(text, list(encoding.ids), list(encoding.offsets))
    ids = _ids_from_call(tokenizer, text, add_special_tokens=False)
    return _with_offsets(text, ids, None)


def content_token_ids(tokenizer: object, text: str) -> list[int]:
    """Token IDs after the tokenizer's own normalizer, without special tokens."""
    return tokenize_content(tokenizer, text).ids


def _with_offsets(text: str, ids: list[int], raw_offsets: object) -> TokenizedText:
    if raw_offsets is None:
        return TokenizedText(ids, None, None, "tokenizer_did_not_provide_offsets")
    parsed = _parse_offsets(text, ids, raw_offsets)
    if parsed is None:
        return TokenizedText(ids, None, None, "offsets_do_not_match_request_text")
    return TokenizedText(ids, parsed, "unicode_code_points", None)


def _parse_offsets(
    text: str,
    ids: list[int],
    raw_offsets: object,
) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(raw_offsets, (list, tuple)) or len(raw_offsets) != len(ids):
        return None
    limit = len(text)
    parsed: list[tuple[int, int]] = []
    for item in raw_offsets:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        start, end = item
        if isinstance(start, bool) or isinstance(end, bool):
            return None
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if start < 0 or end < start or end > limit:
            return None
        parsed.append((start, end))
    return tuple(parsed)


def _observation(
    index: int,
    span: tuple[int, int],
    tokenized: TokenizedText,
    score: float,
) -> WindowObservation:
    start, end = span
    char_start = None
    char_end = None
    if tokenized.offsets is not None and start < end:
        char_start = tokenized.offsets[start][0]
        char_end = tokenized.offsets[end - 1][1]
        if char_end < char_start:
            char_start = None
            char_end = None
    return WindowObservation(index, start, end, char_start, char_end, score)


def _ids_from_call(tokenizer: object, text: str, *, add_special_tokens: bool) -> list[int]:
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


def discover_special_tokens(tokenizer: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Learn the special-token wrapper from the tokenizer, not from constants."""
    probe = "coverage-probe"
    full = _call_ids(tokenizer, probe, add_special_tokens=True)
    content = _ids_from_call(tokenizer, probe, add_special_tokens=False)
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
    return _ids_from_call(tokenizer, text, add_special_tokens=add_special_tokens)


def _batches(windows: Sequence[list[int]], batch_size: int) -> list[list[list[int]]]:
    return [list(windows[start : start + batch_size]) for start in range(0, len(windows), batch_size)]

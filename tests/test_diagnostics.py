import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from prompt_guard.scanner import Scanner, WindowPlan, discover_special_tokens
from prompt_guard.windowing import content_token_spans
from tests.conftest import FixedBackend, StubTokenizer, auth_header, make_app, make_settings


class RowBackend:
    def __init__(self, rows: list[tuple[float, float]]) -> None:
        self.rows = [np.array(row, dtype=np.float64) for row in rows]
        self.calls = 0
        self.batch_sizes: list[int] = []

    def predict_logits(self, input_ids: np.ndarray, attention_mask: np.ndarray, should_stop) -> np.ndarray:
        del attention_mask, should_stop
        count = input_ids.shape[0]
        selected = [self.rows[(self.calls + offset) % len(self.rows)] for offset in range(count)]
        self.calls += count
        self.batch_sizes.append(count)
        return np.stack(selected)


class MappedTokenizer(StubTokenizer):
    def __init__(self, ids: list[int], offsets: list[tuple[int, int]] | None) -> None:
        self._ids = ids
        self._offsets = offsets

    def encode_content(self, text: str) -> dict[str, object]:
        del text
        body: dict[str, object] = {"input_ids": self._ids}
        if self._offsets is not None:
            body["offsets"] = self._offsets
        return body


def _scanner(tokenizer: object, backend: object, **overrides: object) -> Scanner:
    settings = make_settings(window_size=8, window_overlap=2, window_batch_size=2, **overrides)
    prefix, suffix = discover_special_tokens(tokenizer)
    capacity = settings.window_size - len(prefix) - len(suffix)
    return Scanner(
        tokenizer,
        backend,
        WindowPlan(
            capacity=capacity,
            overlap=settings.window_overlap,
            step=capacity - settings.window_overlap,
            prefix=prefix,
            suffix=suffix,
            batch_size=settings.window_batch_size,
            max_windows=settings.max_windows,
            max_input_tokens=settings.max_input_tokens,
            pad_id=0,
            injection_index=1,
        ),
    )


def test_default_response_omits_diagnostics() -> None:
    app, _backend = make_app()
    with TestClient(app) as client:
        response = client.post("/v1/scan", headers=auth_header(), json={"text": "hello"})
    assert response.status_code == 200
    assert "diagnostics" not in response.json()
    assert "return_windows" not in response.json()


def test_windows_come_from_the_same_pass_and_keep_order() -> None:
    rows = [(0.0, 0.0), (0.0, math.log(4.0)), (math.log(9.0), 0.0)]
    backend = RowBackend(rows)
    app, _backend = make_app(backend, window_size=8, window_overlap=2, window_batch_size=2, max_windows=8)
    with TestClient(app) as client:
        plain = client.post("/v1/scan", headers=auth_header(), json={"text": "abcdefghijklmn"})
        diagnosed = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "abcdefghijklmn", "return_windows": True},
        )
    assert plain.status_code == 200
    assert "diagnostics" not in plain.json()
    body = diagnosed.json()
    diagnostics = body["diagnostics"]
    scores = [window["injection_score"] for window in diagnostics["windows"]]
    assert scores == pytest.approx([0.5, 0.8, 0.1])
    assert body["injection_score"] == pytest.approx(max(scores))
    assert diagnostics["winning_window_index"] == 1
    assert diagnostics["aggregation"] == "max"
    assert diagnostics["representation"] == "raw_text_v1"
    assert [window["index"] for window in diagnostics["windows"]] == [0, 1, 2]
    assert diagnostics["windows"][0]["content_token_end"] > diagnostics["windows"][1]["content_token_start"]
    assert backend.batch_sizes == [2, 1, 2, 1]
    assert "abcdefghijklmn" not in diagnosed.text


def test_tie_uses_the_lowest_index() -> None:
    scanner = _scanner(StubTokenizer(), RowBackend([(0.0, 0.0), (0.0, 0.0)]))
    result = scanner.scan("abcdefghij", lambda: False)
    assert result.windows[0].injection_score == result.windows[1].injection_score
    assert result.winning_window_index == 0


def test_character_offsets_are_code_points_and_do_not_search_text() -> None:
    text = "e\u0301x"
    tokenizer = MappedTokenizer([11, 22], [(0, 1), (2, 3)])
    scanner = _scanner(tokenizer, FixedBackend((0.0, math.log(4.0))), max_windows=4)
    result = scanner.scan(text, lambda: False)
    assert result.offset_unit == "unicode_code_points"
    assert result.char_offset_reason is None
    assert result.windows[0].char_start == 0
    assert result.windows[0].char_end == 3
    assert text[0:1] == "e"


def test_bad_offsets_keep_token_intervals_without_character_spans() -> None:
    tokenizer = MappedTokenizer([11, 22], [(0, 99)])
    scanner = _scanner(tokenizer, FixedBackend())
    result = scanner.scan("ab", lambda: False)
    assert result.char_offset_reason == "offsets_do_not_match_request_text"
    assert result.offset_unit is None
    window = result.windows[0]
    assert window.char_start is None
    assert window.char_end is None
    assert window.content_token_start == 0
    assert window.content_token_end == 2
    assert 0.0 <= window.injection_score <= 1.0


def test_return_windows_must_be_boolean_and_errors_stay_score_free() -> None:
    app, _backend = make_app()
    with TestClient(app) as client:
        rejected = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "return_windows": "yes"},
        )
        failed = client.post("/v1/scan", headers=auth_header(), json={"text": "", "return_windows": True})
    assert rejected.status_code == 422
    assert "diagnostics" not in rejected.json()
    assert "flagged" not in failed.json()
    assert "injection_score" not in failed.json()


def test_diagnostics_cover_the_configured_window_maximum() -> None:
    spans = content_token_spans(14, capacity=6, overlap=2)
    assert len(spans) == 3
    assert spans == [(0, 6), (4, 10), (8, 14)]
    backend = RowBackend([(0.0, 0.0)] * 3)
    scanner = _scanner(
        MappedTokenizer(list(range(14)), [(index, index + 1) for index in range(14)]),
        backend,
        max_windows=3,
        max_input_tokens=14,
    )
    result = scanner.scan("a" * 14, lambda: False)
    assert len(result.windows) == 3
    assert result.windows[-1].content_token_end == 14
    assert result.windows[-1].char_end == 14
    oversized = _scanner(StubTokenizer(), FixedBackend(), max_windows=2, max_input_tokens=100)
    with pytest.raises(Exception) as caught:
        oversized.scan("a" * 14, lambda: False)
    assert caught.value.code == "too_many_windows"

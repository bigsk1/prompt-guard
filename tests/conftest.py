"""Shared fakes. No model download."""

import threading
import time

import numpy as np
from pydantic import SecretStr

from prompt_guard.app import create_app
from prompt_guard.config import Settings
from prompt_guard.engine import Engine
from prompt_guard.errors import scan_timeout
from prompt_guard.info import LoadedInfo
from prompt_guard.scanner import Scanner, WindowPlan, discover_special_tokens

API_KEY = "test-api-key-0001"


class StubTokenizer:
    model_max_length = 8192
    pad_token_id = 0

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False, padding: bool = False, **kwargs: object) -> dict[str, list[int]]:
        ids = [ord(character) for character in text]
        if add_special_tokens:
            ids = [2, *ids, 1]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FixedBackend:
    def __init__(self, logits: tuple[float, float] = (0.0, 0.0), delay: float = 0.0) -> None:
        self.logits = np.array(logits, dtype=np.float64)
        self.delay = delay
        self.calls = 0

    def predict_logits(self, input_ids: np.ndarray, attention_mask: np.ndarray, should_stop) -> np.ndarray:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if should_stop():
            raise scan_timeout()
        return np.tile(self.logits, (input_ids.shape[0], 1))


class GateBackend:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def predict_logits(self, input_ids: np.ndarray, attention_mask: np.ndarray, should_stop) -> np.ndarray:
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=5)
        if should_stop():
            raise scan_timeout()
        return np.zeros((input_ids.shape[0], 2), dtype=np.float64)


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "api_key": SecretStr(API_KEY),
        "window_size": 32,
        "window_overlap": 4,
        "window_batch_size": 1,
        "max_windows": 8,
        "max_input_tokens": 500,
        "max_request_bytes": 4096,
        "deadline_ms": 2000,
        "queue_size": 2,
        "default_threshold": 0.5,
    }
    values.update(overrides)
    return Settings(**values)


def make_info(settings: Settings) -> LoadedInfo:
    return LoadedInfo(
        model_id=settings.model_id,
        revision=settings.model_revision,
        model_artifact="onnx/model_quantized.onnx",
        model_sha256="abc",
        tokenizer_sha256="def",
        backend="onnx",
        device="cpu",
        quantization="dynamic-int8",
        providers=("CPUExecutionProvider",),
        labels=((0, "SAFE"), (1, "INJECTION")),
        injection_index=1,
        default_threshold=settings.default_threshold,
        window_size=settings.window_size,
        window_overlap=settings.window_overlap,
        content_tokens_per_window=settings.window_size - 2,
        step=settings.window_size - 2 - settings.window_overlap,
        special_tokens=2,
        context_limit=8192,
        max_request_bytes=settings.max_request_bytes,
        max_input_tokens=settings.max_input_tokens,
        max_windows=settings.max_windows,
        deadline_ms=settings.deadline_ms,
        queue_size=settings.queue_size,
        window_batch_size=settings.window_batch_size,
        onnx_intra_op_threads=None,
        runtime_version="test",
        gpu_name=None,
    )


def make_app(backend: object | None = None, **overrides: object):
    settings = make_settings(**overrides)
    tokenizer = StubTokenizer()
    prefix, suffix = discover_special_tokens(tokenizer)
    capacity = settings.window_size - len(prefix) - len(suffix)
    scanner = Scanner(
        tokenizer,
        backend or FixedBackend(),
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
    engine = Engine(queue_size=settings.queue_size, scanner=scanner, info=make_info(settings))
    return create_app(settings, engine), backend


def auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}

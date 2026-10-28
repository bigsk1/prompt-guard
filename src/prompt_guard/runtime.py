"""Load one pinned checkpoint and build the scanner.

The pytorch backend is imported lazily so the CPU image can omit torch.
"""

import logging
from pathlib import Path

from transformers import AutoTokenizer

from prompt_guard import INJECTION_INDEX
from prompt_guard.config import Settings
from prompt_guard.backends.onnx_backend import OnnxBackend
from prompt_guard.errors import FatalConfigError, ScanError
from prompt_guard.info import LoadedInfo
from prompt_guard.model_store import (
    ONNX_FILES,
    PYTORCH_WEIGHTS,
    ensure_files,
    files_for_runtime,
    read_label_config,
    sha256_file,
)
from prompt_guard.scanner import Scanner, WindowPlan, discover_special_tokens

log = logging.getLogger(__name__)

try:
    from prompt_guard.backends.pytorch_backend import PytorchBackend
except ImportError as exc:
    if "torch" not in str(exc):
        raise
    PytorchBackend = None  # type: ignore[misc, assignment]


def load_scanner(settings: Settings) -> tuple[Scanner, LoadedInfo]:
    if not settings.api_key_configured():
        raise FatalConfigError(
            "PROMPT_GUARD_API_KEY must be set to a random value of at least 16 characters."
        )
    snapshot = ensure_files(
        model_id=settings.model_id,
        revision=settings.model_revision,
        cache_dir=settings.cache_path(),
        filenames=files_for_runtime(settings.runtime, settings.onnx_variant),
    )
    id2label, _label2id, max_position = read_label_config(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    try:
        prefix, suffix = discover_special_tokens(tokenizer)
    except ScanError as exc:
        raise FatalConfigError(exc.message) from exc
    context_limit = _context_limit(tokenizer.model_max_length, max_position)
    if settings.window_size > context_limit:
        raise FatalConfigError(
            f"window_size {settings.window_size} exceeds the model context limit {context_limit}."
        )
    special = len(prefix) + len(suffix)
    capacity = settings.window_size - special
    if capacity < 1:
        raise FatalConfigError("window_size does not leave room for content tokens.")
    if settings.window_overlap >= capacity:
        raise FatalConfigError(
            f"window_overlap {settings.window_overlap} must be smaller than "
            f"the content capacity {capacity}."
        )
    injection_index = id2label_index(id2label)
    backend, info_bits = _load_backend(settings, snapshot)
    plan = WindowPlan(
        capacity=capacity,
        overlap=settings.window_overlap,
        step=capacity - settings.window_overlap,
        prefix=prefix,
        suffix=suffix,
        batch_size=settings.window_batch_size,
        max_windows=settings.max_windows,
        max_input_tokens=settings.max_input_tokens,
        pad_id=int(tokenizer.pad_token_id),
        injection_index=injection_index,
    )
    scanner = Scanner(tokenizer, backend, plan)
    _warmup(scanner)
    artifact, quantization, providers, version, gpu_name, threads = info_bits
    info = LoadedInfo(
        model_id=settings.model_id,
        revision=settings.model_revision,
        model_artifact=artifact,
        model_sha256=sha256_file(snapshot / artifact),
        tokenizer_sha256=sha256_file(snapshot / "tokenizer.json"),
        backend=settings.runtime,
        device=settings.device if settings.runtime == "pytorch" else "cpu",
        quantization=quantization,
        providers=providers,
        labels=tuple((int(index), name) for index, name in sorted(id2label.items(), key=lambda item: int(item[0]))),
        injection_index=injection_index,
        default_threshold=settings.default_threshold,
        window_size=settings.window_size,
        window_overlap=settings.window_overlap,
        content_tokens_per_window=capacity,
        step=capacity - settings.window_overlap,
        special_tokens=special,
        context_limit=context_limit,
        max_request_bytes=settings.max_request_bytes,
        max_input_tokens=settings.max_input_tokens,
        max_windows=settings.max_windows,
        deadline_ms=settings.deadline_ms,
        queue_size=settings.queue_size,
        window_batch_size=settings.window_batch_size,
        onnx_intra_op_threads=threads,
        runtime_version=version,
        gpu_name=gpu_name,
    )
    return scanner, info


def id2label_index(id2label: dict[str, str]) -> int:
    matches = [int(index) for index, name in id2label.items() if name == "INJECTION"]
    if matches != [INJECTION_INDEX]:
        raise FatalConfigError("INJECTION must be label index 1.")
    return matches[0]


def _context_limit(model_max_length: int, max_position_embeddings: int) -> int:
    limit = int(model_max_length)
    if limit > 1_000_000_000:
        limit = max_position_embeddings
    return min(limit, max_position_embeddings)


def _load_backend(settings: Settings, snapshot: Path) -> tuple[object, tuple]:
    if settings.runtime == "onnx":
        artifact = ONNX_FILES[settings.onnx_variant]
        backend = OnnxBackend(str(snapshot / artifact), settings.onnx_intra_op_threads)
        quantization = "dynamic-int8" if settings.onnx_variant == "quantized" else "fp32"
        return backend, (
            artifact,
            quantization,
            backend.providers,
            backend.version,
            None,
            backend.intra_op_threads,
        )
    if PytorchBackend is None:
        raise FatalConfigError(
            "PyTorch is not installed. Use the GPU image, or the ONNX CPU runtime."
        )
    backend = PytorchBackend(str(snapshot), settings.device, settings.dtype)
    return backend, (
        PYTORCH_WEIGHTS,
        backend.quantization,
        backend.providers,
        backend.torch_version,
        backend.gpu_name,
        None,
    )


def _warmup(scanner: Scanner) -> None:
    try:
        scanner.scan("hello", lambda: False)
    except Exception as exc:
        raise FatalConfigError("Model warmup failed.") from exc

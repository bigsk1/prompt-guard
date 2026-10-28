"""Loaded-runtime description returned by GET /v1/model."""

from dataclasses import dataclass

from prompt_guard import AGGREGATION, CONTRACT_VERSION


@dataclass(frozen=True)
class LoadedInfo:
    model_id: str
    revision: str
    model_artifact: str
    model_sha256: str
    tokenizer_sha256: str
    backend: str
    device: str
    quantization: str
    providers: tuple[str, ...]
    labels: tuple[tuple[int, str], ...]
    injection_index: int
    default_threshold: float
    window_size: int
    window_overlap: int
    content_tokens_per_window: int
    step: int
    special_tokens: int
    context_limit: int
    max_request_bytes: int
    max_input_tokens: int
    max_windows: int
    deadline_ms: int
    queue_size: int
    window_batch_size: int
    onnx_intra_op_threads: int | None
    runtime_version: str | None
    gpu_name: str | None

    def public_dict(self) -> dict[str, object]:
        return {
            "contract_version": CONTRACT_VERSION,
            "model": {
                "id": self.model_id,
                "revision": self.revision,
            },
            "artifacts": {
                "model": {"path": self.model_artifact, "sha256": self.model_sha256},
                "tokenizer": {"path": "tokenizer.json", "sha256": self.tokenizer_sha256},
            },
            "runtime": {
                "backend": self.backend,
                "device": self.device,
                "quantization": self.quantization,
                "providers": list(self.providers),
                "version": self.runtime_version,
                "gpu_name": self.gpu_name,
                "inference_workers": 1,
            },
            "labels": [
                {"index": index, "name": name} for index, name in self.labels
            ],
            "injection_label": "INJECTION",
            "injection_index": self.injection_index,
            "threshold": self.default_threshold,
            "window": {
                "size": self.window_size,
                "overlap": self.window_overlap,
                "aggregation": AGGREGATION,
                "size_includes_special_tokens": True,
                "content_tokens_per_window": self.content_tokens_per_window,
                "step": self.step,
                "special_tokens": self.special_tokens,
                "batch_size": self.window_batch_size,
            },
            "limits": {
                "max_request_bytes": self.max_request_bytes,
                "max_input_tokens": self.max_input_tokens,
                "max_windows": self.max_windows,
                "deadline_ms": self.deadline_ms,
                "queue_size": self.queue_size,
                "context_limit": self.context_limit,
                "onnx_intra_op_threads": self.onnx_intra_op_threads,
            },
            "caveats": [
                "Scores are model estimates, not calibrated guarantees of safety.",
                "flagged means injection_score >= threshold. The threshold is not an authorization policy.",
                "A scan that is not flagged does not make the content trusted and does not authorize tool execution.",
                "complete coverage means every input token was scored, not that every attack is detectable.",
            ],
        }

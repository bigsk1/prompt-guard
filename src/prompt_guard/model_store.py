"""Download pinned model files without executing repository Python."""

import hashlib
import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from prompt_guard import EXPECTED_ID2LABEL, EXPECTED_LABEL2ID
from prompt_guard.errors import FatalConfigError

log = logging.getLogger(__name__)

TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
ONNX_FILES = {
    "quantized": "onnx/model_quantized.onnx",
    "fp32": "onnx/model.onnx",
}
PYTORCH_WEIGHTS = "model.safetensors"


def ensure_files(
    *,
    model_id: str,
    revision: str,
    cache_dir: Path,
    filenames: tuple[str, ...],
) -> Path:
    """Return the snapshot directory for the pinned revision.

    Cached files are reused with no network check. Missing files are downloaded.
    Python files in the model repository are not requested.
    """
    snapshot: Path | None = None
    for filename in filenames:
        path = Path(
            _download(
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                filename=filename,
            )
        )
        resolved = _snapshot_dir(path, filename)
        if resolved.name != revision:
            raise FatalConfigError(
                f"Downloaded {filename} resolved to commit {resolved.name}, expected {revision}."
            )
        if snapshot is None:
            snapshot = resolved
        elif snapshot != resolved:
            raise FatalConfigError("Model files resolved to different snapshot directories.")
    if snapshot is None:
        raise FatalConfigError("No model files were requested.")
    return snapshot


def files_for_runtime(runtime: str, onnx_variant: str) -> tuple[str, ...]:
    if runtime == "onnx":
        artifact = ONNX_FILES[onnx_variant]
        return TOKENIZER_FILES + (artifact,)
    if runtime == "pytorch":
        return TOKENIZER_FILES + (PYTORCH_WEIGHTS,)
    raise FatalConfigError(f"Unknown runtime {runtime}.")


def read_label_config(snapshot: Path) -> tuple[dict[str, str], dict[str, int], int]:
    config_path = snapshot / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FatalConfigError("config.json could not be read.") from exc
    id2label = {str(key): str(value) for key, value in dict(raw.get("id2label", {})).items()}
    label2id_raw = dict(raw.get("label2id", {}))
    label2id = {str(key): int(value) for key, value in label2id_raw.items()}
    if id2label != EXPECTED_ID2LABEL or label2id != EXPECTED_LABEL2ID:
        raise FatalConfigError(
            "Label mapping is not SAFE=0 and INJECTION=1. Refusing to score with this checkpoint."
        )
    max_position = raw.get("max_position_embeddings")
    if not isinstance(max_position, int) or max_position < 8:
        raise FatalConfigError("config.json is missing max_position_embeddings.")
    return id2label, label2id, max_position


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(*, model_id: str, revision: str, cache_dir: Path, filename: str) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        return hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except Exception:
        log.info("downloading %s at revision %s", filename, revision)
        return hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )


def _snapshot_dir(file_path: Path, filename: str) -> Path:
    return file_path.parents[len(Path(filename).parts) - 1]

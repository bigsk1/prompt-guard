# AGENTS.md

This file is for a later coding agent. Read it before changing the scanner, the model pin, or the container.

## What this is

`prompt-guard` is a single-process FastAPI service. It loads [Horizon-Labs/prompt-injection-guard-base](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base) once and scores text sent by other programs. The caller applies its own allow or block policy. This service must not grow a policy engine.

The code in this repository is MIT. The checkpoint is Apache-2.0 and is downloaded, not committed.

## Model pin

`DEFAULT_MODEL_REVISION` in `src/prompt_guard/__init__.py` is a full commit SHA. `Settings.model_revision` rejects branches and tags. Startup refuses to score unless `config.json` says `SAFE=0` and `INJECTION=1`.

To move the pin:

1. Read the model card for label names, `max_position_embeddings`, ONNX filenames, and the tokenizer normalizer.
2. Change `DEFAULT_MODEL_REVISION` and the example YAML to the new full SHA.
3. Run `PROMPT_GUARD_RUN_INTEGRATION=1 pytest tests/integration`.
4. Confirm window IDs still match the tokenizer's overflowing-token encoding, obfuscated strings still share token IDs with their plain form, and ONNX stays within 0.05 absolute probability of the PyTorch reference on the examples in that test.
5. Do not relax the label check to keep a mismatched checkpoint loading.

The current pin is `f95005bbceac345aafec1620719857c6202eabd5` (model card v2.2, ModernBERT, context 8192, labels SAFE / INJECTION).

## Layout

| Path | Role |
| --- | --- |
| `src/prompt_guard/app.py` | Routes, auth, body limit, error shape. |
| `src/prompt_guard/engine.py` | One inference thread and a bounded queue. |
| `src/prompt_guard/scanner.py` | Tokenize, window, softmax, max. |
| `src/prompt_guard/windowing.py` | Coverage math. |
| `src/prompt_guard/runtime.py` | Download, label check, backend selection, warmup. |
| `src/prompt_guard/model_store.py` | Hugging Face download of an allow-list of files. |
| `src/prompt_guard/backends/onnx_backend.py` | CPU ONNX Runtime. Can abort a run at the deadline. |
| `src/prompt_guard/backends/pytorch_backend.py` | Reference CPU and CUDA GPU. Optional import. |
| `compose.yaml` | CPU image, ONNX quantized. |
| `compose.gpu.yaml` | Override: PyTorch CUDA float16. |
| `Dockerfile` | Targets `cpu` and `gpu`. |

## Invariants

- `trust_remote_code` stays false. Never download or import `*.py` from the model repo. The tokenizer's bundled normalizer lives in `tokenizer.json` and must be the one that runs.
- Do not log, store, or echo submitted text, `source`, thresholds paired with text, or the API key. `repr(Settings)` hides the key.
- `return_windows` is opt-in and must reuse the scores from the same inference pass. Character offsets come from the tokenizer. Do not recover them by decoding tokens and searching, and do not log window text.
- Do not fetch URLs, open caller paths, or shell out during a scan.
- Do not add a batch route, chat route, policy route, or job queue. Version 1 is one text in, one result or an error.
- A successful body has `status: ok`, `coverage.complete: true`, and `coverage.truncated: false`. A timeout, a size rejection, or an exception must not include `flagged` or `injection_score`.
- `flagged` is `injection_score >= threshold` and nothing else. The server default and the per-request threshold are not an authorization policy.
- `injection_score` is the max INJECTION softmax over windows, using the verified label index.
- Window size includes special tokens discovered from the tokenizer, not hardcoded BOS/EOS ids. It must be `<= min(tokenizer.model_max_length, max_position_embeddings)`. Overlap is content-token overlap, the same value the model card passes as tokenizer `stride`.
- The published tokenizer has truncation enabled at 1024. `content_token_ids` calls `no_truncation()` before encoding. Do not remove that. A long document must be windowed or rejected, never silently cut. Oversized inputs return 413.
- One process, one loaded model, one worker thread. Raising Uvicorn workers duplicates the weights. The GPU image is built for that reason: a 5060 Ti has 16 GB, and a second process still wastes it.
- The deadline covers queue wait, tokenization, and every window. Expired queued jobs are dropped. ONNX sets `RunOptions.terminate` when the deadline passes. PyTorch checks between windows; one forward pass is not killed.
- `source` is parsed and discarded. It must not reach `Scanner.scan`.
- Health routes stay unauthenticated and must not grow config, revision, or threshold fields.

## Runtime matrix

| Image | `RUNTIME` | `DEVICE` | Weights | Quantization reported |
| --- | --- | --- | --- | --- |
| `compose.yaml` | `onnx` | `cpu` | `onnx/model_quantized.onnx` or `onnx/model.onnx` | `dynamic-int8` or `fp32` |
| `compose.gpu.yaml` | `pytorch` | `cuda` | `model.safetensors` | actual parameter dtype, normally `fp16` |

ONNX on CUDA is intentionally absent. The 5060 Ti needs PyTorch built with CUDA 12.8 (`torch==2.11.0` from the `cu128` index), which has `sm_120` kernels. Older CUDA wheels fail with "no kernel image".

`dynamic-int8` means the publisher's `onnx/model_quantized.onnx` (int8 gather / embedding quantization). `/v1/model` also returns the artifact path and sha256. If you change which file loads, change the reported quantization in the same place.

Startup downloads only the allow-listed files, prefers the local snapshot for the pinned SHA, and retries transient download errors. A bad label map, a window that does not fit, a missing GPU, or a missing API key is fatal and is not retried. The process stays up so `/health/live` still answers; `/health/ready` stays 503.

## API contract

Contract version is `1.0` in `src/prompt_guard/__init__.py`.

Authentication is `Authorization: Bearer`. Comparison hashes both strings before `compare_digest`. Health and `GET /` do not require it.

Error JSON is `{request_id, status: "error", error: {code, message, retryable}}`.

Codes: `unauthorized`, `invalid_request`, `request_too_large`, `input_too_long`, `too_many_windows`, `queue_full`, `not_ready`, `scan_timeout`, `internal_error`.

## Config precedence

1. Constructor arguments (tests).
2. `PROMPT_GUARD_*` environment variables, including variables injected by Compose.
3. `.env` when running outside Compose.
4. YAML at `PROMPT_GUARD_CONFIG`.
5. Field defaults.

`queue_size` is the number of requests waiting, not counting the one running. Python's `queue.Queue` treats `maxsize <= 0` as infinite, so the setting rejects values below 1.

## Containers

- CPU base: `python:3.12.12-slim-bookworm`. GPU base: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`.
- The process user is uid 1001. The named volume is created from an image directory owned by that uid.
- Root filesystem is read-only. Model files go to `/models`. `HOME` and scratch go to the `/tmp` tmpfs.
- Compose health checks `/health/live` so a first-time download is not killed. Callers should wait on `/health/ready`.
- Do not bake the weights into the image. Startup pull plus the volume is the cache.
- Do not `pip install` the project directly from a BuildKit bind mount. Those mounts are read-only, and setuptools needs to write `egg-info`. Copy the mounted tree to a writable directory first, as the Dockerfile does.

## Tests

Unit tests use a stub tokenizer and a fake backend. They must stay free of network and of the real checkpoint.

`tests/test_invariants.py` fails if someone sets `trust_remote_code=True` or adds URL fetching to the package.

Integration tests are skipped unless `PROMPT_GUARD_RUN_INTEGRATION=1`. They are the regression net for a model-pin change. The agreement tolerance is 0.05 absolute on the injection probability, and the flagged bit at 0.5 and 0.8 must agree. Tighten that only after measuring a new checkpoint; do not widen it to hide a real mismatch.

## Commits

Co-author trailers for the repository owner must use the GitHub noreply address `32640296+bigsk1@users.noreply.github.com`. Do not write a personal email address into a commit, a config file, a log, or this document.

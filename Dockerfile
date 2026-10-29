# syntax=docker/dockerfile:1

# CPU target is the default. The GPU target is for an RTX 5060 Ti (sm_120) or
# another CUDA 12.8 GPU. PyTorch cu128 is what carries the Blackwell kernels.

FROM python:3.12.12-slim-bookworm AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_HUB_CACHE=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=error \
    HOME=/tmp \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN python -m venv /opt/venv && mkdir -p /tmp/pkg-src

# Bind mounts are read-only. setuptools writes egg-info into the source tree,
# so copy the project to a writable directory before installing it.
RUN --mount=type=bind,source=pyproject.toml,target=/tmp/pkg-src/pyproject.toml \
    --mount=type=bind,source=src,target=/tmp/pkg-src/src \
    --mount=type=cache,target=/root/.cache/pip \
    cp -a /tmp/pkg-src /tmp/pkg \
    && pip install --no-compile /tmp/pkg \
    && rm -rf /tmp/pkg

RUN groupadd --system --gid 1001 appgroup \
    && useradd --system --uid 1001 --gid 1001 --home-dir /tmp --no-create-home appuser \
    && mkdir -p /models \
    && chown 1001:1001 /models

USER 1001:1001
EXPOSE 8080

LABEL org.opencontainers.image.source="https://github.com/bigsk1/prompt-guard" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="prompt-guard" \
      org.opencontainers.image.description="Prompt injection scanning API"

ENTRYPOINT ["python", "-m", "prompt_guard"]


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_HUB_CACHE=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=error \
    HOME=/tmp \
    PATH="/opt/venv/bin:${PATH}" \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ca-certificates \
    && python3 -m venv /opt/venv \
    && mkdir -p /tmp/pkg-src

# Bind mounts are read-only. setuptools writes egg-info into the source tree,
# so copy the project to a writable directory before installing it.
RUN --mount=type=bind,source=pyproject.toml,target=/tmp/pkg-src/pyproject.toml \
    --mount=type=bind,source=src,target=/tmp/pkg-src/src \
    --mount=type=cache,target=/root/.cache/pip \
    cp -a /tmp/pkg-src /tmp/pkg \
    && pip install --no-compile /tmp/pkg \
    && rm -rf /tmp/pkg

# Install CUDA torch after the app so this step cannot be replaced by a CPU wheel.
# cu128 wheels include the sm_120 kernels required by the RTX 5060 Ti.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

RUN groupadd --system --gid 1001 appgroup \
    && useradd --system --uid 1001 --gid 1001 --home-dir /tmp --no-create-home appuser \
    && mkdir -p /models \
    && chown 1001:1001 /models

USER 1001:1001
EXPOSE 8080

LABEL org.opencontainers.image.source="https://github.com/bigsk1/prompt-guard" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="prompt-guard" \
      org.opencontainers.image.description="Prompt injection scanning API with CUDA"

ENTRYPOINT ["python", "-m", "prompt_guard"]

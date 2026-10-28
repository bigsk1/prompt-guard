"""Runtime configuration.

Environment variables prefixed with PROMPT_GUARD_ override a YAML file at
PROMPT_GUARD_CONFIG, which overrides the built-in defaults. The API key is a
secret and is omitted from repr().
"""

import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from prompt_guard import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROMPT_GUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(default=SecretStr(""))
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    model_cache_dir: str = "/models"
    runtime: Literal["onnx", "pytorch"] = "onnx"
    device: Literal["cpu", "cuda"] = "cpu"
    onnx_variant: Literal["quantized", "fp32"] = "quantized"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    window_size: int = 2048
    window_overlap: int = 512
    window_batch_size: int = 4
    max_windows: int = 32
    max_input_tokens: int = 32768
    max_request_bytes: int = 1_048_576
    deadline_ms: int = 30_000
    default_threshold: float = 0.5
    queue_size: int = 8
    retry_after_seconds: int = 1
    onnx_intra_op_threads: int = 0
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    tls_certfile: str | None = None
    tls_keyfile: str | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Init and environment win. YAML is only a fallback for keys they omit.
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        yaml_path = os.environ.get("PROMPT_GUARD_CONFIG")
        if yaml_path:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    @field_validator("tls_certfile", "tls_keyfile", mode="before")
    @classmethod
    def empty_path_is_none(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    @field_validator("model_revision")
    @classmethod
    def revision_is_pinned_sha(cls, value: str) -> str:
        if _COMMIT_SHA.fullmatch(value) is None:
            raise ValueError("model_revision must be a full 40-character commit SHA")
        return value

    @model_validator(mode="after")
    def check_ranges(self) -> "Settings":
        if self.window_size < 8:
            raise ValueError("window_size must be at least 8")
        if self.window_overlap < 0:
            raise ValueError("window_overlap must be >= 0")
        if self.window_overlap >= self.window_size:
            raise ValueError("window_overlap must be smaller than window_size")
        if self.window_batch_size < 1:
            raise ValueError("window_batch_size must be >= 1")
        if self.max_windows < 1:
            raise ValueError("max_windows must be >= 1")
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1")
        if self.max_request_bytes < 1:
            raise ValueError("max_request_bytes must be >= 1")
        if self.deadline_ms < 1:
            raise ValueError("deadline_ms must be >= 1")
        if not 0.0 <= self.default_threshold <= 1.0:
            raise ValueError("default_threshold must be between 0 and 1")
        if self.queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be >= 0")
        if self.onnx_intra_op_threads < 0:
            raise ValueError("onnx_intra_op_threads must be >= 0")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.runtime == "onnx" and self.device != "cpu":
            raise ValueError(
                "The ONNX runtime in this service uses the CPU. "
                "Use runtime=pytorch and device=cuda for the GPU."
            )
        if (self.tls_certfile is None) != (self.tls_keyfile is None):
            raise ValueError("Set both tls_certfile and tls_keyfile, or neither.")
        return self

    def revealed_api_key(self) -> str:
        return self.api_key.get_secret_value()

    def api_key_configured(self) -> bool:
        return len(self.revealed_api_key()) >= 16

    def cache_path(self) -> Path:
        return Path(self.model_cache_dir)


def load_settings() -> Settings:
    yaml_path = os.environ.get("PROMPT_GUARD_CONFIG")
    if yaml_path and not Path(yaml_path).is_file():
        raise FileNotFoundError(f"PROMPT_GUARD_CONFIG does not exist: {yaml_path}")
    return Settings()

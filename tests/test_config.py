import pytest
from pydantic import SecretStr, ValidationError

from prompt_guard import DEFAULT_MODEL_REVISION
from prompt_guard.config import Settings, load_settings


def test_default_revision_is_a_commit_sha() -> None:
    assert len(DEFAULT_MODEL_REVISION) == 40
    Settings(model_revision=DEFAULT_MODEL_REVISION)


def test_branch_revision_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(model_revision="main")


def test_onnx_gpu_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(runtime="onnx", device="cuda")


def test_api_key_is_hidden_from_repr() -> None:
    settings = Settings(api_key=SecretStr("test-api-key-0001"))
    assert "test-api-key-0001" not in repr(settings)


def test_environment_overrides_yaml(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "default_threshold: 0.2\nwindow_size: 1024\nwindow_overlap: 128\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPT_GUARD_CONFIG", str(path))
    monkeypatch.setenv("PROMPT_GUARD_DEFAULT_THRESHOLD", "0.9")
    settings = load_settings()
    assert settings.default_threshold == 0.9
    assert settings.window_size == 1024
    assert settings.window_overlap == 128


def test_short_api_key_is_not_configured() -> None:
    assert Settings(api_key=SecretStr("short")).api_key_configured() is False
    assert Settings(api_key=SecretStr("test-api-key-0001")).api_key_configured() is True

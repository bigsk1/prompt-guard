from pathlib import Path


def test_service_does_not_enable_remote_code_or_fetch_inputs() -> None:
    root = Path("src/prompt_guard")
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "trust_remote_code=True" not in text
    assert "trust_remote_code = True" not in text
    for banned in ("urlopen", "httpx", "requests.get", "subprocess", "os.system"):
        assert banned not in text


def test_torch_is_not_a_required_dependency() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    dependencies = project.split("[project.optional-dependencies]", maxsplit=1)[0]
    assert "torch" not in dependencies

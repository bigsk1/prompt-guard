import json

import pytest

from prompt_guard.errors import FatalConfigError
from prompt_guard.model_store import read_label_config
from prompt_guard.runtime import id2label_index


def test_injection_index_is_class_one() -> None:
    assert id2label_index({"0": "SAFE", "1": "INJECTION"}) == 1


def test_swapped_labels_are_rejected(tmp_path) -> None:
    with pytest.raises(FatalConfigError):
        id2label_index({"0": "INJECTION", "1": "SAFE"})
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
                "label2id": {"LABEL_0": 0, "LABEL_1": 1},
                "max_position_embeddings": 8192,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FatalConfigError):
        read_label_config(tmp_path)


def test_expected_labels_pass(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "id2label": {"0": "SAFE", "1": "INJECTION"},
                "label2id": {"SAFE": 0, "INJECTION": 1},
                "max_position_embeddings": 8192,
            }
        ),
        encoding="utf-8",
    )
    id2label, label2id, limit = read_label_config(tmp_path)
    assert id2label["1"] == "INJECTION"
    assert label2id["SAFE"] == 0
    assert limit == 8192

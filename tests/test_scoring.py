import math

import numpy as np
import pytest

from prompt_guard.scoring import injection_score, softmax


def test_equal_logits_are_one_half() -> None:
    assert injection_score(np.array([[0.0, 0.0]]), 1) == pytest.approx(0.5)


def test_injection_index_selects_class_one() -> None:
    logits = np.array([[0.0, math.log(4.0)]])
    assert injection_score(logits, 1) == pytest.approx(0.8)
    assert softmax(logits)[0, 0] == pytest.approx(0.2)


def test_max_across_batch() -> None:
    logits = np.array([[0.0, math.log(4.0)], [5.0, 0.0]])
    assert injection_score(logits, 1) == pytest.approx(0.8)


def test_non_finite_score_is_rejected() -> None:
    with pytest.raises(ValueError):
        injection_score(np.array([[math.nan, 0.0]]), 1)

"""Map classifier logits to an INJECTION probability."""

import math

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Logits are shaped [batch, labels]."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape [batch, labels]")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def injection_score(logits: np.ndarray, injection_index: int) -> float:
    """Maximum INJECTION-class softmax probability across the batch."""
    probabilities = softmax(logits)
    if injection_index < 0 or injection_index >= probabilities.shape[1]:
        raise ValueError("injection index is outside the label dimension")
    score = float(np.max(probabilities[:, injection_index]))
    if not math.isfinite(score):
        raise ValueError("model returned a non-finite score")
    return score

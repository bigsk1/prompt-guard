import pytest

from prompt_guard.windowing import content_is_covered, plan_windows, window_count


@pytest.mark.parametrize("count", range(0, 80))
def test_windows_cover_every_token(count: int) -> None:
    capacity = 10
    overlap = 3
    step = capacity - overlap
    tokens = list(range(count))
    windows = plan_windows(tokens, capacity=capacity, overlap=overlap, prefix=(9,), suffix=(8,))
    assert content_is_covered(
        tokens,
        windows,
        prefix_len=1,
        suffix_len=1,
        capacity=capacity,
        step=step,
    )
    assert len(windows) == window_count(count, capacity, step)
    assert all(window[0] == 9 and window[-1] == 8 for window in windows)
    assert all(len(window) <= capacity + 2 for window in windows)


def test_overlap_must_advance() -> None:
    with pytest.raises(ValueError):
        plan_windows([1, 2, 3], capacity=4, overlap=4, prefix=(), suffix=())

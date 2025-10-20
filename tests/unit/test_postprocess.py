# FILE: tests/unit/test_postprocess.py
import pytest
from rice2025 import postprocess as pp


def test_majority_simple():
    assert pp.majority_vote([1,1,2]) == 1


def test_majority_tie():
    # tie between 1 and 2 -> deterministic pick smaller -> 1
    assert pp.majority_vote([1,2]) == 1


def test_average_label():
    assert pp.average_label([1,2,3]) == pytest.approx(2.0)


def test_postprocess_empty_raises():
    with pytest.raises(ValueError):
        pp.majority_vote([])

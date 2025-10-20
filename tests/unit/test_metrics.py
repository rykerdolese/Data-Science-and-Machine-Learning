# FILE: tests/unit/test_metrics.py
import pytest
from rice2025 import metrics as m


def test_euclidean_simple():
    assert pytest.approx(m.euclidean([0,0],[3,4])) == 5


def test_manhattan_simple():
    assert m.manhattan([1,2],[4,1]) == 4


def test_metrics_length_mismatch():
    with pytest.raises(ValueError):
        m.euclidean([1,2],[1,2,3])

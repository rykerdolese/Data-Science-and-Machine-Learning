import pytest
from rice2025 import preprocess as pp


def test_normalize_basic():
    X = [[3, 4], [0, 0]]
    out = pp.normalize(X)
    assert pytest.approx(out[0][0]) == 3 / 5
    assert out[1] == [0.0, 0.0]


def test_normalize_none_raises():
    with pytest.raises(ValueError):
        pp.normalize(None)


def test_scale_minmax_constant_col():
    X = [[1, 2], [1, 4], [1, 6]]
    scaled = pp.scale_minmax(X, (0, 1))
    # first column constant -> all 0.5
    assert all(s[0] == 0.5 for s in scaled)


def test_train_test_split_length_mismatch():
    with pytest.raises(ValueError):
        pp.train_test_split([[1],[2]], [1])


def test_train_test_split_empty():
    X_train, X_test, y_train, y_test = pp.train_test_split([], [], test_size=0.5)
    assert X_train == [] and X_test == []
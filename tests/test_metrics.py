from mlpackage.metrics import accuracy_score
import pytest

def test_accuracy_perfect():
    assert accuracy_score([1, 0, 1], [1, 0, 1]) == 1.0

def test_accuracy_half():
    assert accuracy_score([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5

def test_accuracy_mismatched_length():
    with pytest.raises(ValueError):
        accuracy_score([1, 0, 1], [1, 0])

# Edge cases
def test_accuracy_score_mismatched_lengths():
    y_true = [1, 0, 1]
    y_pred = [1, 0]
    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)



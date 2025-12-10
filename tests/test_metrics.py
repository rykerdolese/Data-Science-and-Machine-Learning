from mlpackage.metrics import accuracy_score
import pytest

def test_accuracy_perfect():
    """
    Test accuracy_score on perfectly matching labels.

    Checks
    ------
    - Accuracy is 1.0 when predictions exactly match the true labels.
    """
    assert accuracy_score([1, 0, 1], [1, 0, 1]) == 1.0

def test_accuracy_half():
    """
    Test accuracy_score on partially correct predictions.

    Checks
    ------
    - Accuracy is 0.5 when half of the predictions are correct.
    """
    assert accuracy_score([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5

def test_accuracy_mismatched_length():
    """
    Test accuracy_score with mismatched input lengths.

    Checks
    ------
    - Passing true and predicted labels of different lengths raises ValueError.
    """
    with pytest.raises(ValueError):
        accuracy_score([1, 0, 1], [1, 0])

# Edge cases
def test_accuracy_score_mismatched_lengths():
    """
    Edge case test: mismatched lengths again.

    Checks
    ------
    - Ensures ValueError is consistently raised for different-length inputs.
    """
    y_true = [1, 0, 1]
    y_pred = [1, 0]
    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)




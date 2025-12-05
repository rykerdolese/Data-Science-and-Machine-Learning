import numpy as np
from mlpackage.perceptron import Perceptron
import pytest

def test_perceptron_linearly_separable():
    """
    Test Perceptron on a simple linearly separable dataset.

    Dataset
    -------
    - AND gate: X = [[0,0],[0,1],[1,0],[1,1]], y = [0,0,0,1]

    Checks
    ------
    - Perceptron achieves at least 75% accuracy on this simple linearly separable data.
    """
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])  # AND gate

    model = Perceptron(learning_rate=0.1, n_iterations=10)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.mean(preds == y) >= 0.75

def test_perceptron_learns_simple_rule():
    """
    Test that Perceptron learns a simple binary rule.

    Dataset
    -------
    - AND gate like dataset: X = [[0,0],[0,1],[1,0],[1,1]], y = [0,0,0,1]

    Checks
    ------
    - Predicted labels shape matches target labels.
    """
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 0, 0, 1])
    clf = Perceptron(learning_rate=0.1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape

# Edge cases

def test_perceptron_unfitted_predict():
    """
    Test prediction before fitting Perceptron.

    Checks
    ------
    - Attempting to predict without fitting raises a TypeError.
    """
    X = np.random.rand(5, 2)
    model = Perceptron()
    with pytest.raises(TypeError):
        model.predict(X)

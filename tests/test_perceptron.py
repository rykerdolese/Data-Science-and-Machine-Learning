import numpy as np
from mlpackage.perceptron import Perceptron
import pytest

def test_perceptron_linearly_separable():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])  # AND gate

    model = Perceptron(learning_rate=0.1, n_iterations=10)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.mean(preds == y) >= 0.75

def test_perceptron_learns_simple_rule():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 0, 0, 1])
    clf = Perceptron(learning_rate=0.1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape

# Edge cases

def test_perceptron_unfitted_predict():
    X = np.random.rand(5, 2)
    model = Perceptron()
    with pytest.raises(TypeError):
        model.predict(X)
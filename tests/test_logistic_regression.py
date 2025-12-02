import numpy as np
import pytest
from mlpackage.logistic_regression import LogisticRegression

def test_logistic_regression_basic():
    # Simple separable dataset
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    preds = model.predict(X)

    # Should predict roughly correct classes
    assert set(preds).issubset({0, 1})
    assert np.mean(preds == y) > 0.8

def test_logistic_regression_basic_fit_predict():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})

def test_logistic_regression_probability_output():
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    model = LogisticRegression()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert np.all((probs >= 0) & (probs <= 1))

# Edge cases
def test_logistic_regression_invalid_shape():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=9)  # mismatched length
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_logistic_regression_unfitted_predict():
    model = LogisticRegression()
    X = np.random.rand(3, 2)
    with pytest.raises(TypeError):
        model.predict(X)
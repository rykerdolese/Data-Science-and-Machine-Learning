import numpy as np
import pytest
from mlpackage.logistic_regression import LogisticRegression

def test_logistic_regression_basic():
    """
    Test LogisticRegression on a simple linearly separable dataset.

    Checks
    ------
    - Predicted labels are either 0 or 1.
    - Overall accuracy is greater than 80%.
    """
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    preds = model.predict(X)

    assert set(preds).issubset({0, 1})
    assert np.mean(preds == y) > 0.8

def test_logistic_regression_basic_fit_predict():
    """
    Test basic fit and predict functionality.

    Checks
    ------
    - Predictions have the same shape as input labels.
    - Predictions are either 0 or 1.
    """
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})

def test_logistic_regression_probability_output():
    """
    Test that predict_proba outputs valid probabilities.

    Checks
    ------
    - Predicted probabilities are in the range [0, 1].
    """
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    model = LogisticRegression()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert np.all((probs >= 0) & (probs <= 1))

# Edge cases
def test_logistic_regression_invalid_shape():
    """
    Test fitting with mismatched feature and label shapes.

    Checks
    ------
    - Fitting with X and y of different lengths raises ValueError.
    """
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=9)  # mismatched length
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_logistic_regression_unfitted_predict():
    """
    Test predicting before the model is fitted.

    Checks
    ------
    - Attempting to predict without fitting raises TypeError.
    """
    model = LogisticRegression()
    X = np.random.rand(3, 2)
    with pytest.raises(TypeError):
        model.predict(X)

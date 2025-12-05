import numpy as np
import pytest
from mlpackage.supervised_learning.linear_regression import LinearRegression

def test_linear_regression_perfect_fit():
    """
    Test LinearRegression predictions on perfectly linear data.

    Checks
    ------
    - Predicted values match the expected linear relationship within a small tolerance.
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])  # y = 2x
    model = LinearRegression()
    model.fit(X, y)
    
    preds = model.predict(np.array([[5], [6]]))
    assert np.allclose(preds, [10, 12], atol=1e-6)

def test_linear_regression_rmse_zero_on_perfect_data():
    """
    Test that RMSE is effectively zero on perfectly linear data.

    Checks
    ------
    - RMSE should be below 1e-6 when predictions perfectly match targets.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])  # y = x
    model = LinearRegression()
    model.fit(X, y)
    
    assert model.rmse(X, y) < 1e-6

def test_linear_regression_r_squared_perfect():
    """
    Test that R² score is 1.0 on perfectly linear data.

    Checks
    ------
    - R² score should be very close to 1.0.
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression()
    model.fit(X, y)
    
    r2 = model.R_squared(X, y)
    assert np.isclose(r2, 1.0, atol=1e-6)

def test_linear_regression_r_squared_noisy():
    """
    Test R² score on noisy linear data.

    Checks
    ------
    - R² score should be high (between 0.8 and 1.0) even with noise.
    """
    rng = np.random.default_rng(42)
    X = np.arange(1, 21).reshape(-1, 1)
    y = 3 * X.flatten() + 5 + rng.normal(0, 1, size=20)
    
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.R_squared(X, y)
    assert 0.8 <= r2 <= 1.0

# Edge cases
def test_linear_regression_empty_data():
    """
    Test that fitting on empty data raises ValueError.

    Checks
    ------
    - Fitting with empty feature matrix or target array raises ValueError.
    """
    model = LinearRegression()
    X = np.array([]).reshape(0, 2)
    y = np.array([])
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_linear_regression_unfitted_predict():
    """
    Test that predicting before fitting raises AttributeError.

    Checks
    ------
    - Attempting to predict without fitting the model first raises AttributeError.
    """
    model = LinearRegression()
    X = np.random.rand(5, 2)
    with pytest.raises(AttributeError):
        model.predict(X)

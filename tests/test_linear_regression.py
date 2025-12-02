import numpy as np
import pytest
from src.mlpackage import LinearRegression

def test_linear_regression_perfect_fit():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])  # y = 2x
    model = LinearRegression()
    model.fit(X, y)
    
    preds = model.predict(np.array([[5], [6]]))
    assert np.allclose(preds, [10, 12], atol=1e-6)

def test_linear_regression_rmse_zero_on_perfect_data():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])  # y = x
    model = LinearRegression()
    model.fit(X, y)
    
    assert model.rmse(X, y) < 1e-6

def test_linear_regression_r_squared_perfect():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression()
    model.fit(X, y)
    
    r2 = model.R_squared(X, y)
    assert np.isclose(r2, 1.0, atol=1e-6)

def test_linear_regression_r_squared_noisy():
    rng = np.random.default_rng(42)
    X = np.arange(1, 21).reshape(-1, 1)
    y = 3 * X.flatten() + 5 + rng.normal(0, 1, size=20)
    
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.R_squared(X, y)
    assert 0.8 <= r2 <= 1.0

# Edge cases
def test_linear_regression_empty_data():
    model = LinearRegression()
    X = np.array([]).reshape(0, 2)
    y = np.array([])
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_linear_regression_unfitted_predict():
    model = LinearRegression()
    X = np.random.rand(5, 2)
    with pytest.raises(AttributeError):
        model.predict(X)
import numpy as np
import pytest
from src.mlpackage import DecisionTreeRegressor  # adjust import as needed

def test_decision_tree_regressor_perfect_split():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[0.5], [2.5]]))
    assert np.allclose(preds, np.array([0.0, 1.0]))

def test_decision_tree_regressor_single_value():
    X = np.array([[1], [2], [3]])
    y = np.array([5, 5, 5])
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[0], [10]]))
    assert np.allclose(preds, 5)

def test_decision_tree_regressor_max_depth():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = DecisionTreeRegressor(max_depth=0)
    tree.fit(X, y)
    
    preds = tree.predict(X)
    # With depth=0, all predictions should be the mean of y
    expected_mean = np.mean(y)
    assert np.allclose(preds, expected_mean)

def test_decision_tree_regressor_predict_unseen_values():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[10], [-5]]))
    assert preds.shape == (2,)
    assert np.all(preds >= 0) and np.all(preds <= 1)

# Edge cases
def test_decision_tree_regressor_no_data():
    X = np.array([]).reshape(0, 2)
    y = np.array([])
    tree = DecisionTreeRegressor()
    with pytest.raises(ValueError):
        tree.fit(X, y)

def test_decision_tree_regressor_unfitted_predict():
    tree = DecisionTreeRegressor()
    X = np.random.rand(3, 2)
    with pytest.raises(AttributeError):
        tree.predict(X)

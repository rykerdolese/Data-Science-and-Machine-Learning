import numpy as np
import pytest
from src.mlpackage import DecisionTreeRegressor  # adjust import as needed

def test_decision_tree_regressor_perfect_split():
    """
    Test that the DecisionTreeRegressor correctly predicts on a dataset
    that can be perfectly split.

    The tree should output the exact target values for points between
    the training samples.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[0.5], [2.5]]))
    assert np.allclose(preds, np.array([0.0, 1.0]))

def test_decision_tree_regressor_single_value():
    """
    Test that the DecisionTreeRegressor handles a dataset with a single
    repeated target value.

    All predictions should return the same single value.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([5, 5, 5])
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[0], [10]]))
    assert np.allclose(preds, 5)

def test_decision_tree_regressor_max_depth():
    """
    Test that the DecisionTreeRegressor respects the max_depth parameter.

    When max_depth=0, all predictions should return the mean of the targets.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = DecisionTreeRegressor(max_depth=0)
    tree.fit(X, y)
    
    preds = tree.predict(X)
    # With depth=0, all predictions should be the mean of y
    expected_mean = np.mean(y)
    assert np.allclose(preds, expected_mean)

def test_decision_tree_regressor_predict_unseen_values():
    """
    Test that the DecisionTreeRegressor can predict values outside the
    training range.

    Predictions should return an array of the correct shape, and for
    this dataset, all predicted values should be within the original
    target range [0, 1].
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[10], [-5]]))
    assert preds.shape == (2,)
    assert np.all(preds >= 0) and np.all(preds <= 1)




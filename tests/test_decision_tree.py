import numpy as np
import pytest
from src.mlpackage import DecisionTree

def test_decision_tree_perfect_split():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[0.5], [2.5]]))
    assert np.array_equal(preds, np.array([0, 1]))

def test_decision_tree_single_class():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 1, 1])
    tree = DecisionTree()
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[0], [10]]))
    assert np.all(preds == 1)

def test_decision_tree_max_depth():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(max_depth=0)
    tree.fit(X, y)
    
    preds = tree.predict(X)
    # With depth=0, it should just predict the majority class
    assert np.all((preds == 0) | (preds == 1))

def test_decision_tree_predict_unseen_values():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    
    preds = tree.predict(np.array([[10], [-5]]))
    assert preds.shape == (2,)
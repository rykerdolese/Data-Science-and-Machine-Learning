import numpy as np
import pandas as pd
import pytest
from src.mlpackage import KNN

def test_knn_basic_classification():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = KNN(k=1)
    model.fit(X, y)
    
    preds = model.predict(np.array([[0.5], [2.5]]))
    assert np.array_equal(preds, np.array([0, 1]))

def test_knn_accuracy_perfect():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 1])
    model = KNN(k=1)
    model.fit(X, y)
    assert model.accuracy(X, y) == 1.0

def test_knn_confusion_matrix():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])
    model = KNN(k=1)
    model.fit(X, y)
    
    cm = model.confusion_matrix(X, y)
    assert isinstance(cm, pd.DataFrame)
    assert cm.loc[1, 1] >= 1

def test_knn_k_greater_than_samples():
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model = KNN(k=5)  # larger than number of samples
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 2

# Edge cases
def test_knn_predict_without_fit():
    knn = KNN(k=3)
    X = np.random.rand(3, 2)
    with pytest.raises(AttributeError):
        knn.predict(X)

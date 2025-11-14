import numpy as np
from src.mlpackage import KMeans
import pytest

def test_kmeans_fit_predict():
    X = np.vstack([
        np.random.randn(10, 2) + np.array([0, 0]),
        np.random.randn(10, 2) + np.array([5, 5])
    ])

    kmeans = KMeans(K=2, max_iters=100)
    kmeans.fit(X)
    preds = kmeans.predict(X)

    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})

def test_kmeans_fit_predict_labels():
    X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])
    kmeans = KMeans(K=2)
    kmeans.fit(X)
    preds = kmeans.predict(X)
    assert len(preds) == len(X)
    assert len(np.unique(preds)) <= 2

# Edge cases

def test_kmeans_predict_without_fit():
    kmeans = KMeans(K=2)
    X = np.random.rand(4, 2)
    with pytest.raises(AttributeError):
        kmeans.predict(X)
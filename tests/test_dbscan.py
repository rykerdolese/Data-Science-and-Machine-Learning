"""
Unit tests for the custom DBSCAN implementation.
"""

import numpy as np
import pytest
from src.mlpackage.DBSCAN import DBSCAN

def test_dbscan_basic_clusters():
    """
    Test DBSCAN on a simple two-cluster dataset.
    """
    X = np.array([[0,0],[0,1],[1,0],[1,1],
                  [5,5],[5,6],[6,5],[6,6]])
    db = DBSCAN(eps=1.5, min_samples=2)
    labels = db.fit_predict(X)
    assert set(labels) == {0, 1}
    assert labels.shape[0] == X.shape[0]

def test_dbscan_noise():
    """
    Test DBSCAN labeling points as noise when isolated.
    """
    X = np.array([[0,0],[0,1],[1,0],[10,10]])
    db = DBSCAN(eps=1.0, min_samples=3)
    labels = db.fit_predict(X)
    # Last point should be labeled as noise (-1)
    assert labels[-1] == -1

def test_dbscan_all_noise():
    """
    Test DBSCAN when all points are too far apart (all noise).
    """
    X = np.array([[0,0],[10,10],[20,20]])
    db = DBSCAN(eps=1.0, min_samples=2)
    labels = db.fit_predict(X)
    assert np.all(labels == -1)

def test_dbscan_single_cluster():
    """
    Test DBSCAN when all points are close enough to form one cluster.
    """
    X = np.array([[0,0],[0,0.5],[0.5,0]])
    db = DBSCAN(eps=1.0, min_samples=2)
    labels = db.fit_predict(X)
    assert len(np.unique(labels)) == 1
    assert set(labels) == {0}

def test_dbscan_unfitted_fit_predict_shape():
    """
    Test that fit_predict returns correct shape for any input.
    """
    X = np.random.rand(10,2)
    db = DBSCAN(eps=0.2, min_samples=2)
    labels = db.fit_predict(X)
    assert labels.shape[0] == X.shape[0]

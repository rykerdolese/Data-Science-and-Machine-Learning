import numpy as np
import pandas as pd
import pytest
from mlpackage.supervised_learning.knn import KNN

def test_knn_basic_classification():
    """
    Test basic KNN classification with k=1 on a simple 1D dataset.

    Checks:
    - Predicted labels match expected outputs for clearly separated points.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = KNN(k=1)
    model.fit(X, y)
    
    preds = model.predict(np.array([[0.5], [2.5]]))
    assert np.array_equal(preds, np.array([0, 1]))

def test_knn_accuracy_perfect():
    """
    Test that KNN returns perfect accuracy on a trivially separable dataset.

    Checks:
    - Accuracy method returns 1.0 when predictions perfectly match labels.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 1])
    model = KNN(k=1)
    model.fit(X, y)
    assert model.accuracy(X, y) == 1.0

def test_knn_confusion_matrix():
    """
    Test that KNN generates a valid confusion matrix.

    Checks:
    - Confusion matrix is a pandas DataFrame.
    - The diagonal contains expected counts of correct predictions.
    """
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])
    model = KNN(k=1)
    model.fit(X, y)
    
    cm = model.confusion_matrix(X, y)
    assert isinstance(cm, pd.DataFrame)
    assert cm.loc[1, 1] >= 1

def test_knn_k_greater_than_samples():
    """
    Test KNN behavior when k is larger than the number of samples.

    Checks:
    - Model still returns predictions for all samples.
    - Predictions length matches number of input samples.
    """
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model = KNN(k=5)  # larger than number of samples
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 2

def test_knn_predict_without_fit():
    """
    Test that KNN raises an error when predict is called before fitting.

    Expected behavior:
    - Raises AttributeError if predict is called without prior fit.
    """
    knn = KNN(k=3)
    X = np.random.rand(3, 2)
    with pytest.raises(AttributeError):
        knn.predict(X)


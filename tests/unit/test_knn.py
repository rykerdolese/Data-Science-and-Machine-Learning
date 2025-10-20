# FILE: tests/unit/test_knn.py
import pytest
from rice2025.knn import KNN


def test_knn_input_checks():
    with pytest.raises(ValueError):
        KNN(k=0)
    with pytest.raises(ValueError):
        KNN(mode='bad')


def test_knn_fit_predict_classification():
    X = [[0,0],[1,1],[2,2]]
    y = [0,0,1]
    clf = KNN(k=1, mode='classification')
    with pytest.raises(ValueError):
        clf.fit([], [])
    clf.fit(X, y)
    preds = clf.predict([[0.1, 0.1], [1.9,1.9]])
    assert preds[0] == 0
    assert preds[1] == 1


def test_knn_regression():
    X = [[0],[2],[5]]
    y = [0.0, 2.0, 5.0]
    reg = KNN(k=2, mode='regression')
    reg.fit(X, y)
    preds = reg.predict([[1],[4]])
    # with k=2, first pred avg of 0 and 2 -> 1.0
    assert pytest.approx(preds[0]) == 1.0


def test_knn_not_fitted_raises():
    clf = KNN(k=1)
    with pytest.raises(RuntimeError):
        clf.predict([[0]])





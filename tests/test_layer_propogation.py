import numpy as np
from src.mlpackage import LabelPropagationCustom
import pytest
from sklearn.datasets import make_moons

@pytest.fixture
def simple_dataset():
    """Small synthetic dataset with partial labels"""
    X = np.array([[0,0], [1,0], [0,1], [1,1]])
    y = np.array([0, -1, -1, 1])
    return X, y

@pytest.fixture
def moons_dataset():
    """Semi-labeled moons dataset"""
    X, y_true = make_moons(n_samples=100, noise=0.1, random_state=42)
    y = np.full_like(y_true, -1)
    rng = np.random.default_rng(42)
    for cls in np.unique(y_true):
        idx = np.where(y_true == cls)[0]
        chosen = rng.choice(idx, size=5, replace=False)
        y[chosen] = cls
    return X, y, y_true

def test_basic_propagation(simple_dataset):
    X, y = simple_dataset
    model = LabelPropagationCustom(alpha=0.9, sigma=1.0, max_iter=100)
    model.fit(X, y)
    y_pred = model.predict()
    assert set(y_pred).issubset({0,1})
    # Clamp should keep original labels
    assert y_pred[0] == 0 and y_pred[3] == 1

def test_all_labeled(simple_dataset):
    X, _ = simple_dataset
    y = np.array([0, 0, 1, 1])
    model = LabelPropagationCustom(alpha=0.9, sigma=1.0)
    model.fit(X, y)
    y_pred = model.predict()
    assert np.array_equal(y_pred, y)

def test_no_labeled(simple_dataset):
    X, _ = simple_dataset
    y = np.array([-1, -1, -1, -1])
    model = LabelPropagationCustom(alpha=0.9, sigma=1.0)
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_output_shapes(simple_dataset):
    X, y = simple_dataset
    model = LabelPropagationCustom(alpha=0.9, sigma=1.0)
    model.fit(X, y)
    F = model.F
    y_pred = model.predict()
    assert F.shape[0] == X.shape[0]
    assert F.shape[1] == model.n_classes_
    assert y_pred.shape[0] == X.shape[0]

@pytest.mark.parametrize("alpha", [0.6, 0.8, 0.9])
@pytest.mark.parametrize("sigma", [0.2, 0.4, 0.6])
def test_moons_dataset(alpha, sigma, moons_dataset):
    X, y, y_true = moons_dataset
    model = LabelPropagationCustom(alpha=alpha, sigma=sigma, max_iter=500)
    model.fit(X, y)
    y_pred = model.predict()
    # All predicted labels must be within known classes
    assert set(y_pred).issubset({0,1})
    # Accuracy should be reasonable (>0.5 for semi-supervised)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    assert acc > 0.5

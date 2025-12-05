import numpy as np
from src.mlpackage.unsupervised_learning import LabelPropagationCustom
import pytest
from sklearn.datasets import make_moons

@pytest.fixture
def simple_dataset():
    """
    Small synthetic dataset with partial labels.

    Returns
    -------
    X : np.ndarray of shape (4, 2)
        Feature matrix.
    y : np.ndarray of shape (4,)
        Labels array with -1 for unlabeled points.
    """
    X = np.array([[0,0], [1,0], [0,1], [1,1]])
    y = np.array([0, -1, -1, 1])
    return X, y

@pytest.fixture
def moons_dataset():
    """
    Semi-labeled moons dataset.

    Returns
    -------
    X : np.ndarray of shape (100, 2)
        Feature matrix.
    y : np.ndarray of shape (100,)
        Semi-labeled array, with -1 for unlabeled points.
    y_true : np.ndarray of shape (100,)
        Ground truth labels.
    """
    X, y_true = make_moons(n_samples=100, noise=0.1, random_state=42)
    y = np.full_like(y_true, -1)
    rng = np.random.default_rng(42)
    for cls in np.unique(y_true):
        idx = np.where(y_true == cls)[0]
        chosen = rng.choice(idx, size=5, replace=False)
        y[chosen] = cls
    return X, y, y_true

def test_basic_propagation(simple_dataset):
    """
    Test that label propagation assigns labels correctly on a small dataset.

    Checks
    ------
    - Predicted labels are subset of known classes.
    - Original labeled points remain unchanged after propagation.
    """
    X, y = simple_dataset
    model = LabelPropagationCustom(alpha=0.9, sigma=1.0, max_iter=100)
    model.fit(X, y)
    y_pred = model.predict()
    assert set(y_pred).issubset({0,1})
    # Clamp should keep original labels
    assert y_pred[0] == 0 and y_pred[3] == 1

def test_all_labeled(simple_dataset):
    """
    Test that fully labeled datasets remain unchanged after fitting.

    Checks
    ------
    - Predicted labels equal the original labels.
    """
    X, _ = simple_dataset
    y = np.array([0, 0, 1, 1])
    model = LabelPropagationCustom(alpha=0.9, sigma=1.0)
    model.fit(X, y)
    y_pred = model.predict()
    assert np.array_equal(y_pred, y)

def test_no_labeled(simple_dataset):
    """
    Test that model raises ValueError when no labels are provided.

    Checks
    ------
    - Fitting a dataset with all -1 labels raises ValueError.
    """
    X, _ = simple_dataset
    y = np.array([-1, -1, -1, -1])
    model = LabelPropagationCustom(alpha=0.9, sigma=1.0)
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_output_shapes(simple_dataset):
    """
    Test the shapes of outputs from fit and predict.

    Checks
    ------
    - Label score matrix F has correct shape (n_samples, n_classes).
    - Predicted labels array has correct length.
    """
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
    """
    Test label propagation on a semi-supervised moons dataset with multiple parameters.

    Parameters
    ----------
    alpha : float
        Propagation coefficient.
    sigma : float
        Kernel width parameter.
    moons_dataset : tuple
        Fixture returning (X, y, y_true).

    Checks
    ------
    - All predicted labels are within the known classes.
    - Accuracy exceeds 0.5 for semi-supervised propagation.
    """
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


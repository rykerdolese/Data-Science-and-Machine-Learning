import numpy as np
import pytest
from mlpackage.unsupervised_learning.pca import PCA

def test_pca_fit_transform_shape():
    """
    Test that PCA fit_transform returns the correct shape and attributes.

    Checks
    ------
    - Transformed data has shape (n_samples, n_components)
    - Components matrix has shape (n_components, n_features)
    - Explained variance and explained variance ratio lengths match n_components
    """
    X = np.random.rand(10, 5)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    assert X_pca.shape == (10, 3)
    assert pca.components_.shape == (3, 5)
    assert len(pca.explained_variance_) == 3
    assert len(pca.explained_variance_ratio_) == 3

def test_pca_all_components():
    """
    Test PCA when n_components is None (all components retained).

    Checks
    ------
    - Transformed data shape matches number of original features
    - Components matrix shape matches (n_features, n_features)
    """
    X = np.random.rand(8, 4)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    assert X_pca.shape[1] == 4
    assert pca.components_.shape == (4, 4)


def test_pca_unfitted_transform():
    """
    Test that calling transform on an unfitted PCA raises an AttributeError.

    Checks
    ------
    - Transform before fit raises AttributeError
    """
    X = np.random.rand(5, 2)
    pca = PCA(n_components=1)
    with pytest.raises(AttributeError):
        pca.transform(X)

def test_explained_variance_ratio_sum():
    """
    Test that the explained variance ratio sums to approximately 1.

    Checks
    ------
    - Sum of explained_variance_ratio_ is close to 1
    """
    X = np.random.rand(10, 3)
    pca = PCA()
    pca.fit(X)
    ratio_sum = np.sum(pca.explained_variance_ratio_)
    assert np.isclose(ratio_sum, 1.0, atol=1e-6)


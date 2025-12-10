import numpy as np
import pandas as pd
from mlpackage.preprocess import train_test_split, MinMaxScaler, StandardScaler, OrdinalEncoder
import pytest

def test_train_test_split_shapes():
    """
    Test train_test_split returns correct shapes.

    Checks
    ------
    - Splitting 10 samples with test_size=0.2 produces 8 training samples and 2 test samples.
    - Target arrays have matching lengths.
    """
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert len(y_train) == 8 and len(y_test) == 2

def test_minmax_scaler_range():
    """
    Test MinMaxScaler scales features to [0,1] range.

    Checks
    ------
    - All scaled values are between 0 and 1.
    """
    X = np.array([[1, 2], [3, 4]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.all(X_scaled >= 0) and np.all(X_scaled <= 1)

def test_standard_scaler_mean_std():
    """
    Test StandardScaler standardizes features.

    Checks
    ------
    - Mean of each feature is approximately 0.
    - Standard deviation of each feature is approximately 1.
    """
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)

def test_ordinal_encoder():
    """
    Test OrdinalEncoder transforms categorical features into integer codes.

    Checks
    ------
    - Encoded output shape matches input DataFrame.
    """
    df = pd.DataFrame({
        'edu': ['High', 'Low', 'Medium'],
        'loan': ['Yes', 'No', 'Yes']
    })
    encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High'], ['No', 'Yes']])
    encoded = encoder.fit_transform(df)
    assert encoded.shape == (3, 2)

def test_train_test_split_shapes_random():
    """
    Test train_test_split on random dataset.

    Checks
    ------
    - 100 samples split with test_size=0.2 results in 80 training and 20 test samples.
    """
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_test) == 20
    assert len(X_train) == 80

def test_minmax_scaler_range_single_feature():
    """
    Test MinMaxScaler on single feature array.

    Checks
    ------
    - Scaled values are exactly 0 and 1 for min and max.
    """
    X = np.array([[1], [2], [3]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.isclose(X_scaled.min(), 0)
    assert np.isclose(X_scaled.max(), 1)

def test_standard_scaler_mean_std_single_feature():
    """
    Test StandardScaler on single feature array.

    Checks
    ------
    - Mean approximately 0 and std approximately 1.
    """
    X = np.array([[1], [2], [3]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.isclose(X_scaled.mean(), 0, atol=1e-7)
    assert np.isclose(X_scaled.std(), 1, atol=1e-7)

# Edge cases

def test_standard_scaler_unfit_transform():
    """
    Test StandardScaler raises error if transform is called before fit.

    Checks
    ------
    - Transforming without fitting raises AttributeError.
    """
    scaler = StandardScaler()
    X = np.random.rand(5, 2)
    with pytest.raises(AttributeError):
        scaler.transform(X)



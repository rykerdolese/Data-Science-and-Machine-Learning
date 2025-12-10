import numpy as np
from mlpackage.supervised_learning.multilayer_perceptron import MLP
import pytest

def test_mlp_forward_shape():
    """
    Test the forward pass output shape and probability normalization of MLP.

    Checks
    ------
    - Output shape matches (n_samples, n_output_classes).
    - Softmax probabilities sum to 1 for each sample.
    """
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, size=10)

    model = MLP(layer_dims=[2, 5, 2], actFun_type='tanh', reg_lambda=0.01)
    probs = model.feedforward(X)

    assert probs.shape == (10, 2)
    assert np.allclose(np.sum(probs, axis=1), 1, atol=1e-6)

def test_mlp_forward_pass():
    """
    Test MLP forward pass with improper layer_dims.

    Checks
    ------
    - Using invalid or insufficient layer dimensions raises IndexError during fit/predict.
    """
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, 10)
    with pytest.raises(IndexError):
        mlp = MLP(layer_dims=(5,))
        mlp.fit(X, y)
        preds = mlp.predict(X)
        assert preds.shape == y.shape

def test_mlp_invalid_hidden_layer_sizes():
    """
    Test MLP initialization with invalid hidden layer sizes.

    Checks
    ------
    - Providing non-iterable or invalid layer_dims raises TypeError.
    """
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    with pytest.raises(TypeError):
        MLP(layer_dims="invalid").fit(X, y)

def test_mlp_unfitted_predict():
    """
    Test prediction before fitting the MLP.

    Checks
    ------
    - Attempting to predict without fitting raises IndexError.
    """
    with pytest.raises(IndexError):
        mlp = MLP(layer_dims=(5,))
        X = np.random.rand(4, 2)
        with pytest.raises(IndexError):
            mlp.predict(X)

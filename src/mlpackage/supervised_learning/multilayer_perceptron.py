"""
multilayer_perceptron.py

A minimal NumPy-based implementation of a fully connected neural network (MLP),
including:

- Layer class: linear transformation + activation
- MLP class: multi-layer feedforward network with softmax output
- Forward propagation, backpropagation, cross-entropy loss, and gradient descent

This module is intended for educational use to illustrate how neural networks
operate internally without relying on deep learning frameworks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Layer:
    """
    A single fully connected neural network layer with activation.

    Parameters
    ----------
    input_dim : int
        Number of input features to the layer.
    output_dim : int
        Number of neurons (output features) in the layer.
    actFun_type : str, default='tanh'
        Activation function type. Supports: 'tanh', 'sigmoid', 'relu'.
    seed : int, default=0
        Random seed for reproducibility.

    Attributes
    ----------
    W : np.ndarray of shape (input_dim, output_dim)
        Weight matrix.
    b : np.ndarray of shape (1, output_dim)
        Bias vector.
    z : np.ndarray
        Linear pre-activation values.
    a : np.ndarray
        Activations after applying the activation function.
    """

    def __init__(self, input_dim, output_dim, actFun_type='tanh', seed=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.actFun_type = actFun_type

    def actFun(self, z):
        """
        Apply the activation function.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation input.

        Returns
        -------
        np.ndarray
            Activation output.
        """
        if self.actFun_type == 'tanh':
            return np.tanh(z)
        elif self.actFun_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.actFun_type == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unsupported activation: {self.actFun_type}")

    def diff_actFun(self, z):
        """
        Compute derivative of the activation function.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation input.

        Returns
        -------
        np.ndarray
            Derivative of activation function.
        """
        if self.actFun_type == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.actFun_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif self.actFun_type == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation: {self.actFun_type}")

    def feedforward(self, X):
        """
        Forward pass of this layer.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Activations of this layer.
        """
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a


class MLP:
    """
    A simple multi-layer perceptron (MLP) classifier using softmax output.

    Parameters
    ----------
    layer_dims : list of int
        Layer sizes including input and output. Example: [2, 10, 3].
    actFun_type : str, default='tanh'
        Activation function for hidden layers.
    reg_lambda : float, default=0.01
        L2 regularization strength.
    seed : int, default=0
        Random seed for reproducibility.

    Attributes
    ----------
    layers : list of Layer
        Hidden layers.
    W_out : np.ndarray
        Output layer weights.
    b_out : np.ndarray
        Output layer biases.
    """

    def __init__(self, layer_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.layer_dims = layer_dims
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.layers = []

        # Build hidden layers
        for i in range(len(layer_dims) - 2):
            self.layers.append(Layer(layer_dims[i], layer_dims[i + 1],
                                     actFun_type, seed + i))

        # Output layer (softmax)
        np.random.seed(seed)
        self.W_out = np.random.randn(layer_dims[-2], layer_dims[-1]) / np.sqrt(layer_dims[-2])
        self.b_out = np.zeros((1, layer_dims[-1]))

    def feedforward(self, X):
        """
        Forward pass through the entire network.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class probability predictions (softmax output).
        """
        a = X
        for layer in self.layers:
            a = layer.feedforward(a)

        self.z_out = np.dot(a, self.W_out) + self.b_out
        exp_scores = np.exp(self.z_out - np.max(self.z_out, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        """
        Compute cross-entropy loss with L2 regularization.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            Integer class labels.

        Returns
        -------
        float
            Loss value.
        """
        num_samples = X.shape[0]
        self.feedforward(X)
        data_loss = -np.sum(np.log(self.probs[np.arange(num_samples), y])) / num_samples
        reg_loss = 0.5 * self.reg_lambda * (
            np.sum(self.W_out**2) + sum(np.sum(layer.W**2) for layer in self.layers)
        )
        return data_loss + reg_loss

    def backprop(self, X, y):
        """
        Perform backpropagation to compute gradients.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Class labels.

        Returns
        -------
        grads : list of tuples
            Gradients (dW, db) for each hidden layer.
        dW_out : np.ndarray
            Gradient of output layer weights.
        db_out : np.ndarray
            Gradient of output layer biases.
        """
        num_samples = X.shape[0]
        self.feedforward(X)

        # Output layer gradient
        delta_out = self.probs.copy()
        delta_out[np.arange(num_samples), y] -= 1

        last_hidden = self.layers[-1].a if self.layers else X
        dW_out = np.dot(last_hidden.T, delta_out) / num_samples
        db_out = np.sum(delta_out, axis=0, keepdims=True) / num_samples

        # Backprop through hidden layers
        deltas = delta_out
        grads = []

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_a = self.layers[i - 1].a if i > 0 else X

            backprop_W = self.W_out.T if i == len(self.layers) - 1 else self.layers[i + 1].W.T
            deltas = np.dot(deltas, backprop_W) * layer.diff_actFun(layer.z)

            dW = np.dot(prev_a.T, deltas) / num_samples
            db = np.sum(deltas, axis=0, keepdims=True) / num_samples
            grads.insert(0, (dW, db))

        return grads, dW_out, db_out

    def fit(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        Train the MLP using gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.
        epsilon : float, default=0.01
            Learning rate.
        num_passes : int, default=20000
            Number of gradient descent iterations.
        print_loss : bool, default=True
            Whether to print loss periodically.
        """
        for i in range(num_passes):
            grads, dW_out, db_out = self.backprop(X, y)

            # Regularization on gradients
            dW_out += self.reg_lambda * self.W_out
            for j, (dW, db) in enumerate(grads):
                dW += self.reg_lambda * self.layers[j].W

            # Parameter update
            self.W_out -= epsilon * dW_out
            self.b_out -= epsilon * db_out

            for j, (dW, db) in enumerate(grads):
                self.layers[j].W -= epsilon * dW
                self.layers[j].b -= epsilon * db

            if print_loss and i % 1000 == 0:
                print(f"Iteration {i}, Loss: {self.calculate_loss(X, y)}")

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Predicted integer labels.
        """
        probs = self.feedforward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
            Softmax output probabilities.
        """
        return self.feedforward(X)

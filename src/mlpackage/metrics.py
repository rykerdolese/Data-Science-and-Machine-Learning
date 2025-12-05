"""
metrics.py

A minimal implementation of fundamental classification and regression metrics, including:

- accuracy_score : fraction of correctly predicted labels
- classification_report : precision, recall, and F1-score per class
- rmse : root mean squared error for regression predictions

These functions offer lightweight, NumPy-based alternatives to
sklearn.metrics for educational and experimental purposes.

Typical usage
-------------
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
error = rmse(y_true, y_pred)
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy classification score.

    Accuracy is defined as:
        accuracy = (number of correct predictions) / (total predictions)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels from a classifier.

    Returns
    -------
    float
        Fraction of correctly classified samples.

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    return np.mean(y_true == y_pred)


def classification_report(y_true, y_pred):
    """
    Generate a classification report containing precision, recall, and F1-score per class.

    Precision:
        tp / (tp + fp)

    Recall:
        tp / (tp + fn)

    F1-score:
        2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    dict
        A dictionary mapping each class to a metrics dictionary with keys:
        'precision', 'recall', 'f1-score'.

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.
    """
    from collections import defaultdict

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    classes = np.unique(np.concatenate((y_true, y_pred)))
    report = defaultdict(dict)

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        report[cls]["precision"] = precision
        report[cls]["recall"] = recall
        report[cls]["f1-score"] = f1

    return dict(report)


def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    RMSE is defined as:
        RMSE = sqrt(mean((y_true - y_pred)^2))

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

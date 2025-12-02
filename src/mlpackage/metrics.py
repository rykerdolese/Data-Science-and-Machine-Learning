import numpy as np
def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy classification score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
        The fraction of correctly classified samples.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    return np.mean(y_true == y_pred)

# classification_report function can be added here similarly if needed
def classification_report(y_true, y_pred):
    """
    Generate a classification report.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    report : dict
        A dictionary containing precision, recall, f1-score for each class.
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        report[cls]['precision'] = precision
        report[cls]['recall'] = recall
        report[cls]['f1-score'] = f1

    return dict(report)

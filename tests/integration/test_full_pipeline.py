# tests/integration/test_full_pipeline.py

import pytest
import numpy as np
from rice2025 import preprocess, knn, postprocess

@pytest.mark.integration
def test_full_pipeline_classification():
    # --- Step 1: Create synthetic dataset ---
    np.random.seed(42)
    X = np.random.rand(20, 3)  # 20 samples, 3 features
    y = np.array([0, 1] * 10)  # binary classification

    # --- Step 2: Preprocess data ---
    X_scaled = preprocess.normalize(X)
    X_train, X_test, y_train, y_test = preprocess.train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    # --- Step 3: Train & predict with KNN ---
    model = knn.KNN(k=3, distance_metric="euclidean")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # --- Step 4: Postprocess predictions ---
    final_label = postprocess.majority_label(preds)

    # --- Step 5: Assertions ---
    assert len(preds) == len(y_test), "Predictions and test labels must match in length"
    assert final_label in [0, 1], "Majority label must be a valid class label"
    assert np.all((preds == 0) | (preds == 1)), "Predictions should be binary"

@pytest.mark.integration
def test_full_pipeline_regression():
    # --- Step 1: Create synthetic regression dataset ---
    np.random.seed(123)
    X = np.random.rand(15, 2)
    y = np.random.rand(15) * 10  # continuous target

    # --- Step 2: Preprocess data ---
    X_scaled = preprocess.scale(X)
    X_train, X_test, y_train, y_test = preprocess.train_test_split(
        X_scaled, y, test_size=0.3, random_state=123
    )

    # --- Step 3: Train & predict with KNN ---
    model = knn.KNN(k=4, distance_metric="manhattan")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # --- Step 4: Postprocess predictions ---
    avg_label = postprocess.average_label(preds)

    # --- Step 5: Assertions ---
    assert len(preds) == len(y_test)
    assert isinstance(avg_label, float)
    assert np.isfinite(avg_label), "Average label must be a finite number"
    assert np.all((preds >= 0) & (preds <= 10)), "Predictions within expected range"

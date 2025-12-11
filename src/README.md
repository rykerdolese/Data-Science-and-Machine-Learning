# src

This directory contains the **core implementation of the custom `mlpackage` library**, which provides from-scratch machine learning algorithms modeled after the scikit-learn API. The focus is on clarity, correctness, and learning rather than maximum performance.

---

## Installation & Setup

This project is designed to be installed locally as a Python package.

### Create and activate a virtual environment (recommended)

python -m venv .venv  
source .venv/bin/activate   # macOS / Linux  
.venv\Scripts\activate      # Windows  

### Install required packages

pip install numpy pandas matplotlib pytest  

### Install `mlpackage` in editable mode

From the project root (where `src/` lives):

pip install -e .

This allows you to import modules like:

from mlpackage import LinearRegression  
from mlpackage import PCA
from mlpackage.metrics import accuracy_score  

---

## Design Goals

- Mimic **scikit-learn-style APIs**
- Use only **NumPy and core Python**
- Keep implementations **interpretable and modular**
- Make all models **unit-testable**
- Clearly separate **supervised vs unsupervised** learning

All models follow a consistent interface:
- `fit(X, y)` for training
- `predict(X)` for inference (if applicable)
- `transform(X)` for preprocessing or dimensionality reduction
- `fit_transform(X)` where appropriate

---

## Package Structure

mlpackage/  
- metrics.py  
- preprocess.py  
- supervised_learning/  
- unsupervised_learning/  

Each submodule contains a self-contained implementation with numpy-style docstrings.

---

## Supervised Learning (`mlpackage/supervised_learning`)

Algorithms that learn from labeled data `(X, y)`:

- Linear Regression
- Logistic Regression
- Perceptron
- K-Nearest Neighbors (KNN)
- Decision Tree (Classifier & Regressor)
- Multilayer Perceptron (MLP)

These models support prediction and evaluation using metrics like accuracy and RMSE.

---

## Unsupervised Learning (`mlpackage/unsupervised_learning`)

Algorithms that discover structure without labels:

- K-Means Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- Label Propagation

These models focus on clustering, density estimation, and dimensionality reduction.

---

## Preprocessing & Utilities

preprocess.py:
- train_test_split
- StandardScaler
- MinMaxScaler
- OrdinalEncoder

metrics.py:
- accuracy_score
- rmse
- classification_report

These utilities are intentionally lightweight replacements for sklearn equivalents.

---

## Testing Philosophy

Every algorithm is paired with:
- Shape checks
- Numerical correctness tests
- Edge case tests (unfitted models, invalid inputs)

Tests are written with `pytest` and designed to mirror real-world usage patterns.

---

## Intended Use

This package is intended for:
- Learning how ML algorithms work internally
- Coursework and experimentation
- Replacing sklearn in controlled environments
- Demonstrating algorithmic understanding in interviews or portfolios

It is **not intended for production-scale workloads**, but rather as a clean, understandable reference implementation.



# MLPackage: End-to-End Machine Learning Package

## Overview
`MLPackage` is a **comprehensive end-to-end machine learning framework** designed for educational purposes, experimentation, and rapid prototyping.  
It provides **implementations of supervised, unsupervised, and reinforcement learning algorithms**, along with utilities for preprocessing, evaluation, and visualization.  

Key features include:
- Modular and extensible architecture for adding new algorithms.
- Built-in **data preprocessing** tools (scalers, encoders, train/test split).
- Support for **performance evaluation** (accuracy, confusion matrix, classification reports, RMSE, R², etc.).
- **Visualization helpers** for decision boundaries, loss curves, and clustering results.
- Fully unit-tested and reproducible pipelines.

The package comes with **example datasets** and scripts to demonstrate practical usage.

---

## Project Structure

```bash
.
├── examples
│   ├── Supervised Learning
│   │   ├── Decision Trees
│   │   ├── Ensembles
│   │   ├── KNN
│   │   ├── Linear Regression
│   │   ├── Logistic Regression
│   │   ├── Neural Networks
│   │   └── Perceptron
│   └── Unsupervised Learning
│       ├── DBSCAN
│       ├── K-means
│       ├── Label Propogation
│       ├── PCA
│       └── SVD
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements.txt
├── setup.py
├── src
│   └── mlpackage
│       ├── __init__.py
│       ├── metrics.py
│       ├── preprocess.py
│       ├── supervised_learning
│       └── unsupervised_learning
└── tests
    ├── __init__.py
    ├── test_dbscan.py
    ├── test_decision_tree_classifier.py
    ├── test_decision_tree_regressor.py
    ├── test_kmeans.py
    ├── test_knn.py
    ├── test_layer_propogation.py
    ├── test_linear_regression.py
    ├── test_logistic_regression.py
    ├── test_metrics.py
    ├── test_mlp.py
    ├── test_pca.py
    ├── test_perceptron.py
    └── test_preprocess.py

```
---

## Algorithms Included

### Supervised Learning
- **Linear Regression** — Ordinary Least Squares (OLS) with RMSE and R² evaluation.
- **Logistic Regression** — Batch gradient descent with sigmoid activation, probability outputs, and accuracy metrics.
- **K-Nearest Neighbors (KNN)** — Classic distance-based classifier supporting multiple distance metrics.
- **Decision Trees** — Recursive tree-building using information gain and entropy (classification) or variance reduction (regression).
- **Perceptron** — Single-layer neural network for binary classification.
- **Multi-Layer Perceptron (MLP)** — Feedforward neural network with backpropagation, softmax output, and cross-entropy loss.

### Unsupervised Learning
- **K-Means Clustering** — Centroid-based clustering with inertia computation.
- **DBSCAN** — Density-based clustering for arbitrary-shaped clusters.
- **PCA** — Dimensionality reduction and variance analysis.
- **SVD** — Singular value decomposition for feature extraction and latent representation.

### Reinforcement Learning
- Placeholder directory for RL algorithms (Q-Learning, Policy Gradient, etc.) to be extended in future versions.

---

## Utilities

- **Preprocessing**
  - `MinMaxScaler`, `StandardScaler` — Feature scaling.
  - `OrdinalEncoder` — Convert categorical features to numerical codes.
  - `train_test_split` — Flexible train/test partitioning.
- **Metrics**
  - `accuracy_score`, `classification_report`, RMSE, R².
- **Visualization**
  - Decision boundaries for classifiers.
  - Loss curves for neural networks.
  - Cluster plots for unsupervised learning.

---

## Example Datasets

### Bank Churn
- Features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited`.
- Derived features: TF-IDF of surname, engineered ratios (`Cred_Bal_Sal`, `Bal_sal`, `Tenure_Age`), and one-hot encoded `Country` and `Gender`.

### Loan Data
- Features: `person_age`, `person_gender`, `person_education`, `person_income`, `loan_amnt`, `loan_intent`, `loan_int_rate`, `credit_score`, etc.

### Student Marks
- Features: `number_courses`, `time_study`, `Marks`.

### Titanic
- Features: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

### Wine
- Features: `Alcohol`, `Malic_Acid`, `Ash`, `Ash_Alcanity`, `Magnesium`, `Total_Phenols`, `Flavanoids`, `Color_Intensity`, `Hue`, `OD280`, `Proline`.

---

## Testing

- Unit tests cover:
  - Correct implementation of ML algorithms.
  - Edge cases (empty inputs, invalid parameters).
  - Preprocessing utilities (scalers, encoders, split functions).
  - Metric computation and evaluation functions.
- Run tests with:

```bash
pytest
```
---

## Installation

```bash
git clone <repo_url>
cd MLPackage
pip install -e .
```

---

## Getting Started

```python
from src.mlpackage import LinearRegression, KNN, train_test_split, MinMaxScaler

# Load data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
print("R²:", model.R_squared(X_test_scaled, y_test))
```



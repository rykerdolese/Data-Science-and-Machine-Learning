# MLPackage: End-to-End Machine Learning Package

## Overview
`MLPackage` is a comprehensive end-to-end machine learning framework that provides implementations of supervised, unsupervised, and reinforcement learning algorithms. It is designed for educational purposes, experiments, and rapid prototyping, while also including utilities for preprocessing, evaluation, and visualization.

The package comes with multiple example datasets, fully unit-tested algorithms, and modular code structure to support easy extension.

---

## Project Structure

```bash
├── dl_env/ # Python virtual environment
├── example.py # Sample script demonstrating package usage
├── other/ # Additional unit and integration tests
│ ├── integration/
│ └── unit/
├── pyproject.toml # Project configuration
├── pytest.ini # Pytest configuration
├── README.md # This file
├── Reinforcement Learning/ # Reinforcement learning examples
├── setup.py # Package installation
├── src/
│ ├── mlpackage/ # Core package code
│ ├── mlpackage.egg-info/
│ ├── README.md # Package-specific readme
│ └── rice2025/
├── Supervised Learning/ # Supervised learning examples
│ ├── Decision Trees/
│ ├── Ensembles/
│ ├── KNN/
│ ├── Linear Regression/
│ ├── Logistic Regression/
│ ├── Neural Networks/
│ └── Perceptron/
├── tests/ # Unit tests
│ ├── init.py
│ ├── test_decision_tree.py
│ ├── test_kmeans.py
│ ├── test_knn.py
│ ├── test_linear_regression.py
│ ├── test_logistic_regression.py
│ ├── test_metrics.py
│ ├── test_mlp.py
│ ├── test_perceptron.py
│ └── test_preprocess.py
├── tests.yml # Test configuration
└── Unsupervised Learning/ # Unsupervised learning examples
├── DBSCAN/
├── K-means/
├── PCA/
└── SVD/
```
---

## Algorithms Included

### Supervised Learning
- **Linear Regression** — Ordinary Least Squares (OLS) with RMSE and R² evaluation.
- **Logistic Regression** — Batch gradient descent with sigmoid activation, probability outputs, and accuracy.
- **K-Nearest Neighbors (KNN)** — Classic distance-based classification.
- **Decision Trees** — Recursive tree-building with information gain and entropy.
- **Perceptron** — Single-layer neural network for binary classification.
- **Multi-Layer Perceptron (MLP)** — Feedforward neural network with backpropagation, softmax output, and cross-entropy loss.

### Unsupervised Learning
- **K-Means Clustering** — Centroid-based clustering with inertia computation.
- **DBSCAN** — Density-based clustering.
- **PCA** — Dimensionality reduction.
- **SVD** — Singular value decomposition for feature extraction.

### Reinforcement Learning
- Placeholder directory for RL implementations (to be extended).

---

## Example Datasets

### Bank Churn
- `Surname`, `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited`
- Derived features: `Surname_tfidf_*`, one-hot encoded `Country` and `Gender`, engineered ratios like `Cred_Bal_Sal`, `Bal_sal`, `Tenure_Age`, `Age_Tenure_product`.

### Loan Data
- `person_age`, `person_gender`, `person_education`, `person_income`, `person_emp_exp`, `person_home_ownership`, `loan_amnt`, `loan_intent`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`, `loan_status`.

### Student Marks
- `number_courses`, `time_study`, `Marks`.

### Titanic
- `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

### Wine
- `Alcohol`, `Malic_Acid`, `Ash`, `Ash_Alcanity`, `Magnesium`, `Total_Phenols`, `Flavanoids`, `Nonflavanoid_Phenols`, `Proanthocyanins`, `Color_Intensity`, `Hue`, `OD280`, `Proline`.

---

## Testing

- Unit tests are located in `tests/`.
- Tests cover:
  - Correct implementation of ML algorithms.
  - Edge cases (empty inputs, invalid parameters).
  - Preprocessing utilities (MinMaxScaler, OrdinalEncoder).
- Run tests using:
```bash
pytest
```



# Examples: Machine Learning From Scratch

This folder contains **hands-on implementations and walkthroughs** of core machine learning algorithms built from scratch and applied to real or synthetic datasets.  
Each subfolder includes Jupyter notebooks, datasets, and short READMEs that demonstrate **how and why** each algorithm works.

The examples are organized into **Supervised Learning** and **Unsupervised Learning**, reflecting the two major paradigms in machine learning.

## Supervised Learning

**Supervised learning** uses *labeled data*, meaning each input has a known target output.  
The objective is to learn a function that maps inputs → outputs and generalizes well to unseen data.

### Common Goals
- **Classification** — Predict discrete class labels
- **Regression** — Predict continuous numerical values

### Algorithms in This Folder
- **Linear Regression**  
  Predicts continuous values by fitting a linear relationship between features and targets.

- **Logistic Regression**  
  Performs binary classification using a sigmoid function to model class probabilities.

- **K-Nearest Neighbors (KNN)**  
  A distance-based method that classifies points using the labels of their nearest neighbors.

- **Decision Trees**  
  Hierarchical models that split data into rules based on feature thresholds.

- **Ensembles**  
  Combine multiple models (e.g., bagging or boosting) to improve accuracy and reduce variance.

- **Perceptron**  
  A foundational linear classifier that forms the basis of neural networks.

- **Neural Networks**  
  Multi-layer, non-linear models capable of learning complex patterns.

### Key Characteristics
- Requires labeled training data  
- Evaluated using metrics such as accuracy, RMSE, and R²  
- Common in prediction-focused tasks  

---

## Unsupervised Learning

**Unsupervised learning** works with *unlabeled data*.  
The aim is to identify structure, relationships, or lower-dimensional representations within the data.

### Common Goals
- **Clustering** — Group similar data points
- **Dimensionality Reduction** — Reduce feature space while preserving structure
- **Representation Learning** — Learn informative data encodings

### Algorithms in This Folder
- **K-Means**  
  Groups data by minimizing within-cluster variance.

- **DBSCAN**  
  Density-based clustering that identifies core samples, noise, and arbitrary-shaped clusters.

- **Label Propagation**  
  Semi-supervised approach that spreads label information through a similarity graph.

- **PCA (Principal Component Analysis)**  
  Projects data onto orthogonal directions capturing maximum variance.

- **SVD (Singular Value Decomposition)**  
  Factorizes matrices for compression and feature extraction.

### Key Characteristics
- No labels required  
- Often used for exploratory analysis or preprocessing  
- Useful when data structure is unknown  

---

## Supervised vs. Unsupervised Learning

| Aspect | Supervised Learning | Unsupervised Learning |
|------|---------------------|-----------------------|
| Labels | Required | Not required |
| Output | Explicit predictions | Patterns or structure |
| Common Tasks | Classification, Regression | Clustering, Compression |
| Evaluation | Accuracy, RMSE, R² | Inertia, reconstruction error |
| Typical Use | Predict known outcomes | Discover hidden structure |

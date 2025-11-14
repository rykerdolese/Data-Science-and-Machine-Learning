# Principal Component Analysis

Principal Component Analysis (PCA) is a widely used technique for **dimensionality reduction** and **exploratory data analysis**. As datasets grow in size and complexity, PCA provides a way to simplify the feature space while retaining as much of the original variation as possible. This not only helps with **visualization** but can also improve downstream modeling by reducing **multicollinearity** and noise.

At a high level, PCA works by:

1. **Centering the data** (removing the mean of each feature).
2. **Finding directions (principal components)** along which the data varies the most. These directions are obtained by computing the eigenvectors of the covariance matrix (or via singular value decomposition).
3. **Projecting the data** onto these components, ordered by the amount of variance they explain.

The first few principal components typically capture most of the variation in the dataset, meaning we can often represent high-dimensional data with just 2–3 dimensions for visualization, or with a reduced set of components for modeling.

When applying PCA in practice, there are a few common approaches:

* **Visualization:** Plotting the first two or three principal components to uncover clusters, trends, or outliers in the data.
* **Dimensionality reduction for modeling:** Retaining only the number of components needed to explain a target level of variance (e.g., 90–95%).
* **Model selection:** Using techniques like the **elbow method** to balance complexity and information preserved.

While powerful, PCA has trade-offs: it assumes **linear relationships** in the data, the transformed components may lose **interpretability** compared to the original features, and it can be sensitive to feature scaling.

In `pca_from_scratch.ipynb`, we:

* Use PCA to identify patterns in a churn dataset
* Implement the **elbow method** to select an optimal number of components
* Analyze how much variance is preserved at each step
* Reduce the model feature space to address **multicollinearity**, applying PCA-transformed features in a logistic regression model



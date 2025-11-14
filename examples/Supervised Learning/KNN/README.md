# K-Nearest Neighbors

In the attached notebook `knn_from_scratch.ipynb`, we look at a simple yet effective non-parametric model: K-nearest neighbors (KNN). At its core, to make a prediction for a new observation, KNN works by finding similar observations and predicting the average (or majority for classification) among those similar observations.

KNN is non-parametric because it doesn't have any defined or learned parameters; rather, it finds similar instances and uses those to predict. There aren't any coefficients like in logistic or linear regression. Instead, the model’s performance heavily depends on the choice of distance metric (e.g., Euclidean distance) and the number of neighbors (k).

We train our own KNN algorithm from scratch, applying it to loan application data. Our goal is to predict if the loan was approved. In  the notebook, we dive into the math and intuition -- as well as the overall effectiveness of this simple algorithm.


---
**In this project, we**:

- Implement KNN from scratch to better understand the mechanics behind the algorithm.

- Apply it to loan application data, where the task is to predict whether a loan was approved.

- Explore the intuition and mathematics behind the algorithm, including the role of distance metrics and the effect of varying k.

- Evaluate the strengths and weaknesses of KNN, including its interpretability, sensitivity to scaling, and computational cost on larger datasets.

This notebook is intended as a hands-on introduction to KNN, providing both conceptual understanding and practical implementation.


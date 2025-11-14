# Perceptron

The Perceptron is one of the earliest and simplest models of a neural network, and a foundational building block of many modern deep learning algorithms. At its core, it assigns weights to inputs and produces a classification based on whether the weighted sum exceeds a given threshold (often set to 0). This results in a **linear decision boundary**.

The perceptron learns by updating its weights using **hinge loss**, which penalizes misclassified points and pushes the decision boundary toward better separation. Compared to probabilistic models like logistic regression, the perceptron is more deterministic, focusing on correctly classifying points rather than estimating probabilities.

Despite its simplicity, the perceptron played a crucial role in the development of machine learning, paving the way for more advanced architectures like multi-layer perceptrons and deep neural networks.

In the `perceptron_from_scratch.ipynb` notebook, we:

* Explore the **background, history, and mathematical foundation** of the perceptron
* Perform **exploratory data analysis** on the Titanic dataset, identifying useful features for survival prediction
* **Build our own single-neuron perceptron**, implementing gradient descent for optimization
* **Analyze results and limitations**, discussing when the perceptron works well and when more complex models are needed







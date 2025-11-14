# Neural Networks

**Neural Networks** (often referred to as *multi-layer perceptrons*) are a family of powerful, non-linear supervised learning algorithms. They are inspired by the structure of the human brain, where neurons connect and interact to produce complex behaviors and decisions. While the biological comparison is loose, the idea of interconnected units passing signals forward captures the essence of neural networks.

At a high level, a neural network combines many perceptrons organized into **layers of neurons**. Each neuron computes a weighted combination of its inputs and passes the result through an **activation function**, which introduces non-linearity. By stacking many layers, neural networks can capture complex patterns and relationships in data. In fact, with enough hidden units, a neural network can approximate virtually any non-linear function (*the universal approximation theorem*).

However, this flexibility comes with trade-offs. Neural networks often involve **hundreds, thousands, or even millions of parameters**, which make them computationally expensive to train and challenging to interpret — hence the nickname *“black box models.”* Despite this, their predictive power makes them indispensable in modern machine learning, especially in domains like **computer vision, natural language processing (NLP), and recommendation systems**.

In `neural_networks.ipynb`, we:

* Explain the **underlying math and intuition** behind neural networks
* Explore the **bank churn dataset** to see how a NN can be applied in practice
* **Build our own neural network** using TensorFlow
* Assess model performance and **compare accuracy against simpler algorithms** used earlier to predict churn



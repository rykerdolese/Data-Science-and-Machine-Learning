# Label Propagation Notebook — Explanation

This notebook demonstrates a complete implementation of the Label Propagation algorithm from scratch.

## What the Notebook Does

### 1. Implements a custom `LabelPropagationCustom` class
- Builds an affinity matrix using an RBF kernel.
- Normalizes the matrix to create a transition matrix.
- Propagates labels through iterative updates.
- Supports clamping of labeled nodes.
- Outputs both soft label distributions and final class predictions.

### 2. Generates a synthetic dataset
- Uses `make_moons`, which is ideal for semi-supervised learning.
- Only a few points are labeled; the algorithm spreads labels to the rest.

### 3. Runs the Label Propagation algorithm
- Initializes with partial labels.
- Trains until convergence or a max iteration count.

### 4. Visualizes results
- Ground-truth labels.
- Initial partially labeled dataset.
- Final predictions with confidence shading.

This provides a full educational demonstration of how label propagation works in practice.

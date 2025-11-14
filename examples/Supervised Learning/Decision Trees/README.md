# Decision Trees

In the attached notebook `decision_tree_from_scratch.ipynb`, we explore the intuition, math, and possible uses for decision trees in machine learning. We apply this on the famous Iris dataset, focusing on a binary classification task. Decision trees are a supervised learning algorithm that uses 'True/False' questions to subdivide the data. These 'questions' are used to isolate the response variable classes as best as possible. Metrics such as entropy or gini index are often used for the split criteria. For classification tasks, we predict the most common class at the final node. For regression tasks, we typically average the output variable across the different observations in the terminal node.

### Key Topics Covered in the Notebook:
1. **Intuition Behind Decision Trees**:
   - How decision trees split data using binary questions.
   - The role of metrics like entropy and gini index in determining splits.

2. **Mathematical Foundation**:
   - Detailed explanation of entropy and gini index calculations.
   - How decision trees optimize splits to minimize impurity.

3. **Visualization**:
   - Examples of visualized decision trees to understand the structure and splits.
   - Insights into how the tree isolates classes or predicts continuous values.

4. **Applications**:
   - Use cases for decision trees in classification and regression tasks.
   - Advantages such as interpretability and simplicity.

5. **Limitations**:
   - Overfitting in deep trees and how pruning can address this.
   - Comparisons to ensemble methods like random forests and gradient boosting.

### Why Decision Trees?
Decision trees are very easy to interpret and understand. Often, we can visualize them to see the exact questions a decision tree asks and how well it isolates the classes. They serve as a foundational algorithm in machine learning and are often used as building blocks for more advanced ensemble methods.

For more detailed explanations, examples, and visualizations, refer to the notebook `decision_tree_from_scratch.ipynb`.
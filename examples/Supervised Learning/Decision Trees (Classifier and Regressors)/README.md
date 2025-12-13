# Decision Trees

In the attached notebook `decision_tree_from_scratch.ipynb`, we explore the intuition, math, and possible uses for decision trees in machine learning. We apply this on the famous Iris dataset for a binary classification task and on a **student marks dataset using our DecisionTreeRegressor**, which performs very well in predicting continuous scores.

Decision trees are a supervised learning algorithm that uses 'True/False' questions to subdivide the data. These 'questions' isolate the response variable classes (classification) or predict values (regression). Metrics such as entropy or gini index are often used for the split criteria in classification tasks. For regression tasks, the predicted value at a leaf is typically the **average of the output variable** for the observations that fall in that leaf.

### Key Topics Covered in the Notebook:
1. **Intuition Behind Decision Trees**:
   - How decision trees split data using binary questions.
   - The role of metrics like entropy and gini index in determining splits.
   - How regression trees differ: instead of predicting the most common class, they predict the **mean of the target variable** in the leaf node.

2. **Mathematical Foundation**:
   - Detailed explanation of entropy and gini index calculations for classification.
   - Variance reduction as a criterion for regression trees.
   - How decision trees optimize splits to minimize impurity or variance.

3. **Visualization**:
   - Examples of visualized decision trees to understand the structure and splits.
   - Insights into how the tree isolates classes or predicts continuous values.

4. **Applications**:
   - Use cases for decision trees in classification and regression tasks.
   - Predicting student marks (continuous values) using DecisionTreeRegressor.
   - Advantages such as interpretability and simplicity.

5. **Limitations**:
   - Overfitting in deep trees and how pruning can address this.
   - Comparisons to ensemble methods like random forests and gradient boosting.

### Why Decision Trees?
Decision trees are easy to interpret and visualize. You can see exactly what questions the tree asks and how it isolates classes or predicts continuous values. Regression trees, like our application to student marks, differ from classification trees by **predicting averages of outcomes instead of classes**, making them ideal for continuous targets.

For more detailed explanations, examples, and visualizations, refer to the notebook `decision_tree_from_scratch.ipynb`.

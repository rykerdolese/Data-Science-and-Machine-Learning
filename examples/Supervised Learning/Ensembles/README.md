# Ensemble Models

In the attached notebook `ensembles_from_scratch.ipynb`, we dive into the ensemble structure. We cover bagging, random forests, and boosting. We also briefly touch on stacking at the beginning of the notebook.

For context, ensemble models typically build upon decision trees. They incorporate many (usually smaller) decision trees to make a more accurate prediction. This is typically referred to as an ensemble of 'weak learners.' We cover three primary ways of decision tree ensembles: bagging, random forests, & boosting.

### Key Topics Covered in the Notebook:
1. **Bagging**:
    - Sampling data with replacement, averaging (or majority vote for classification) the prediction over many trees

2. **Random Forest**:
    - Similar to Bagging, creates many trees with resampled data
    - Limits correlation between trees by only sampling a certain number of features for each tree or split

3. **Boosting**:
    - Starts with one decision tree, and each additional one attempts to predict the error of the previous ones
    - The outputs are added to generate a final prediction
    - The trees are sequential rather than independent

4. **Stacking**:
    - Combines predictions from multiple models (not limited to decision trees) using a meta-model for final predictions.

### Additional Insights:
- **Bank Churn Dataset**:
    - We apply ensemble models to predict customer churn using the bank churn dataset.
    - This real-world example demonstrates the practical use of ensemble methods in classification tasks.

- **Custom Random Forest Algorithm**:
    - We implement our own random forest algorithm from scratch, showcasing the inner workings of bagging and feature sampling.
    - This is compared to the `sklearn` implementation to highlight differences in performance and flexibility.

- **Tree Visualization**:
    - We visualize individual decision trees within the ensemble to understand how they contribute to predictions.
    - This helps interpret the model and understand the decision-making process.

- **Mathematical Depth**:
    - The notebook goes deeper into the math behind ensemble methods, including:
        - How bagging reduces variance.
        - The role of feature sampling in random forests.
        - Gradient boosting optimization and loss functions.

### Why Ensemble Models?
Ensemble models are powerful because they combine the strengths of multiple weak learners to create a robust and accurate predictor. They are widely used in both classification and regression tasks due to their ability to handle complex datasets and reduce overfitting.

For detailed explanations, code examples, and visualizations, refer to the notebook `ensembles_from_scratch.ipynb`.

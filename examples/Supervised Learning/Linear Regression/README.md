# Linear Regression

Linear regression is a foundational statistical model used to predict continuous numerical variables. We explore the intuition, math, and assumptions in `lin_reg_from_scratch.ipynb`.

At a high level, linear regression finds the optimal linear combination of explanatory variables to predict a given response variable. This assumes a linear relationship between predictors and the outcome. The “optimal” fit is determined by minimizing the sum of squared differences between the predicted and actual values (the least squares method).

Because of its simplicity, linear regression is often the first model introduced in statistics and machine learning. It provides a clear way to interpret the effect of predictors through their coefficients, but it also relies on several assumptions (linearity, independence, homoscedasticity, and normally distributed errors) that can limit its effectiveness in practice.

In the notebook, we implement linear regression from scratch and use it on real data, building intuition around how the model works, where it succeeds, and where it may fall short.

--- 
**In the `lin_reg_from_scratch.ipynb` notebook, we**:
- Explore the math behind linear rgeression, and the closed form solution through the *normal equations*.
- We perform EDA on the student grades dataset, identifying potential predictors for final grade
- Build a linear regression model from scratch
- Perform residual analsysi to assess the validity of assumptions
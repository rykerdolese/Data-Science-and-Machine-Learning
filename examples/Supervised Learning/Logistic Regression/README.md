# Logistic Regression

Logistic Regression, like Linear Regression, is a fundamental statistical model. It is more suitable for classification than linear regression, which can predict values below 0 or greater than 1 due to it's linear nature. Logistic Regression leverages the sigmoid function, which ranges from 0 to 1, allowing a final probabilitic output.

At a high level, logistic regression estimates the probability that an observation belongs to a given class (e.g., loan approved vs. not approved). A threshold (commonly 0.5) is then applied to convert this probability into a final class prediction.

Because the model is parametric, each explanatory variable has an associated coefficient, which can be interpreted as the effect of that variable on the log-odds of the outcome. This interpretability is one of the main strengths of logistic regression. However, it also relies on assumptions such as linearity of predictors in the log-odds space and independence of observations.

In `log_reg_from_scratch.ipynb`, we dive deeper into the math, intuition, and implementation, applying the model to loan application data to predict whether someone was approved for a given loan.
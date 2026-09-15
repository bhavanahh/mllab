# PROGRAM NO. 3 – ML

**ROLL NO: 56**

## AIM

Implement Ridge and Lasso regression on the Diabetes dataset. Compare the performance of these regularized models with standard linear regression.

## TASKS

* Load and preprocess the dataset.
* Implement Ridge and Lasso regression.
* Tune hyperparameters using cross-validation.
* Compare performance metrics (MSE, R-squared) with standard linear regression.

## ALGORITHM

1. Import the required libraries.
2. Load the Diabetes dataset.
3. Split the data into training and testing sets.
4. Standardize the input features.
5. Train a Linear Regression model.
6. Calculate MSE and R-squared for Linear Regression.
7. Create Ridge and Lasso Regression models.
8. Use cross-validation to find the best alpha value for Ridge and Lasso.
9. Calculate MSE and R-squared for Ridge and Lasso.
10. Compare the performance of all three models.

## PROGRAM

```python
# Import required libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Task 1: Load Dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target


# Task 2: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Task 3: Linear Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)

y_pred_lr = lr_pipeline.predict(X_test)

lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)


# Task 4: Ridge Regression
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

ridge_params = {
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}

ridge_cv = GridSearchCV(
    ridge_pipeline,
    ridge_params,
    cv=5,
    scoring='neg_mean_squared_error'
)

ridge_cv.fit(X_train, y_train)

y_pred_ridge = ridge_cv.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)


# Task 5: Lasso Regression
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(max_iter=10000))
])

lasso_params = {
    'model__alpha': [0.001, 0.01, 0.1, 1, 10]
}

lasso_cv = GridSearchCV(
    lasso_pipeline,
    lasso_params,
    cv=5,
    scoring='neg_mean_squared_error'
)

lasso_cv.fit(X_train, y_train)

y_pred_lasso = lasso_cv.predict(X_test)

lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)


# Task 6: Compare the Models
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],
    "Best Alpha": [
        "-",
        ridge_cv.best_params_['model__alpha'],
        lasso_cv.best_params_['model__alpha']
    ],
    "MSE": [
        round(lr_mse, 2),
        round(ridge_mse, 2),
        round(lasso_mse, 2)
    ],
    "R2 Score": [
        round(lr_r2, 2),
        round(ridge_r2, 2),
        round(lasso_r2, 2)
    ]
})

print("\nPerformance Comparison:")
print(results)

print("\nBest Ridge Alpha:", ridge_cv.best_params_['model__alpha'])
print("Best Lasso Alpha:", lasso_cv.best_params_['model__alpha'])
```

## RESULT

The Diabetes dataset was successfully loaded and preprocessed. Linear Regression, Ridge Regression, and Lasso Regression were implemented. Ridge and Lasso hyperparameters were tuned using 5-fold cross-validation. The models were compared using MSE and R-squared, and the best alpha values were obtained for the regularized models.

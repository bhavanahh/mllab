## LINEAR REGRESSION

### AIM

Implement linear regression with one variable on the California Housing dataset to predict housing prices based on a single feature (average number of rooms per dwelling).

### TASKS

- Load and preprocess the dataset.
- Implement linear regression using both gradient descent and normal equation.
- Evaluate the model performance using metrics such as mean squared error (MSE) and R-squared.
- Visualize the fitted line along the data points.

### ALGORITHM

1. Start
2. Load the California Housing Dataset.
3. Select AveRooms as input and MedHouseVal as output.
4. Split the dataset into training and testing data.
5. Standardize the input feature.
6. Initialize the input and parameters.
7. Apply Gradient Descent.
8. Calculate the parameter using Normal Equation.
9. Predict housing prices.
10. Calculate MSE and R² score.
11. Plot actual data and regression line.
12. Stop

### PROGRAM

```python
# ------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Task 1: Load and Preprocess the Dataset
# ------------------------------------------------------------

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("First Five Rows:")
print(df.head())

X = df[['AveRooms']].values
y = df['MedHouseVal'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# Task 2: Linear Regression using Gradient Descent
# ------------------------------------------------------------

X_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]

theta = np.zeros((2, 1))

learning_rate = 0.01
iterations = 1000
m = len(X_train_scaled)

for i in range(iterations):
    predictions = X_b.dot(theta)
    gradients = (2/m) * X_b.T.dot(predictions - y_train)
    theta = theta - learning_rate * gradients

print("\nGradient Descent Parameters")
print("Intercept:", theta[0][0])
print("Coefficient:", theta[1][0])

X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]
y_pred_gd = X_test_b.dot(theta)


# ------------------------------------------------------------
# Task 3: Linear Regression using Normal Equation
# ------------------------------------------------------------

X_train_ne = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_ne = np.c_[np.ones((len(X_test), 1)), X_test]

theta_ne = np.linalg.pinv(X_train_ne).dot(y_train)

print("\nNormal Equation Parameters")
print("Intercept:", theta_ne[0][0])
print("Coefficient:", theta_ne[1][0])

y_pred_ne = X_test_ne.dot(theta_ne)


# ------------------------------------------------------------
# Task 4: Model Evaluation
# ------------------------------------------------------------

print("\nPerformance")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R^2 Score:", r2_score(y_test, y_pred_gd))


# ------------------------------------------------------------
# Task 5: Visualization
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, alpha=0.5, label="Actual Data")

sorted_index = X_test[:, 0].argsort()

plt.plot(
    X_test[sorted_index],
    y_pred_ne[sorted_index],
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Average Rooms per Household")
plt.ylabel("Median House Value")
plt.title("Linear Regression on California Housing Dataset")
plt.legend()
plt.grid(True)
plt.show()
```

### RESULT

Thus, linear regression was successfully implemented on the California Housing dataset using Gradient Descent and Normal Equation. The model performance was evaluated using MSE and R² score, and the fitted regression line was visualized along with the actual data points.
"""

path = Path("/mnt/data/Linear_Regression_Program_1.md")
path.write_text(md, encoding="utf-8")
print(path)

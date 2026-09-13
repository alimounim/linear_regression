import numpy as np
from data_utils import load_iris_split
from LinearRegression import LinearRegression

# Regularization comparison, using Model 3: sepal length & sepal width -> petal length
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_train[:, [0, 1]]  # sepal length and sepal width
y = X_train[:, 2]       # petal length

REG_STRENGTH = 10.0

# Train an unregularized model
model_plain = LinearRegression(batch_size=32, max_steps=100, regularization=0)
model_plain.fit(X, y)

# Train an identical model, but with L2 regularization added
model_reg = LinearRegression(batch_size=32, max_steps=100, regularization=REG_STRENGTH)
model_reg.fit(X, y)

print("=== Regularization comparison (Model 3: sepal L, sepal W -> petal L) ===")
print(f"Regularization strength used: {REG_STRENGTH}\n")

print("No regularization:")
print(f"  weights: {model_plain.weights.ravel()}")
print(f"  bias:    {model_plain.bias.ravel()}\n")

print("With L2 regularization:")
print(f"  weights: {model_reg.weights.ravel()}")
print(f"  bias:    {model_reg.bias.ravel()}\n")

weight_diff = model_reg.weights - model_plain.weights
bias_diff = model_reg.bias - model_plain.bias

print("Difference (regularized - unregularized):")
print(f"  weight diff: {weight_diff.ravel()}")
print(f"  bias diff:   {bias_diff.ravel()}")
print(f"  weight L2 norm, no reg:   {np.linalg.norm(model_plain.weights):.4f}")
print(f"  weight L2 norm, with reg: {np.linalg.norm(model_reg.weights):.4f}")

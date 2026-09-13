from data_utils import load_iris_split
from LinearRegression import LinearRegression

# Multi-output model: sepal length & sepal width -> petal length & petal width
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_test[:, [0, 1]]  # sepal length and sepal width
y = X_test[:, [2, 3]]  # petal length and petal width

model = LinearRegression()
model.load("regression_multi_weights.npz")

mse = model.score(X, y)
print(f"Multi-output model (sepal L,W -> petal L,W) test MSE: {mse}")

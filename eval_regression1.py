from data_utils import load_iris_split
from LinearRegression import LinearRegression

# Model 1: sepal length -> sepal width
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_test[:, [0]]  # sepal length
y = X_test[:, 1]    # sepal width

model = LinearRegression()
model.load("regression1_weights.npz")

mse = model.score(X, y)
print(f"Model 1 (sepal length -> sepal width) test MSE: {mse}")

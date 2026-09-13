from data_utils import load_iris_split
from LinearRegression import LinearRegression

# Model 2: petal length -> petal width
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_test[:, [2]]  # petal length
y = X_test[:, 3]    # petal width

model = LinearRegression()
model.load("regression2_weights.npz")

mse = model.score(X, y)
print(f"Model 2 (petal length -> petal width) test MSE: {mse}")

from data_utils import load_iris_split
from LinearRegression import LinearRegression

# Model 3: sepal length & sepal width -> petal length
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_test[:, [0, 1]]  # sepal length and sepal width
y = X_test[:, 2]       # petal length

model = LinearRegression()
model.load("regression3_weights.npz")

mse = model.score(X, y)
print(f"Model 3 (sepal length & sepal width -> petal length) test MSE: {mse}")

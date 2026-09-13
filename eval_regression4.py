from data_utils import load_iris_split
from LinearRegression import LinearRegression

# Model 4: petal length & petal width -> sepal length
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_test[:, [2, 3]]  # petal length and petal width
y = X_test[:, 0]       # sepal length

model = LinearRegression()
model.load("regression4_weights.npz")

mse = model.score(X, y)
print(f"Model 4 (petal length & petal width -> sepal length) test MSE: {mse}")

from data_utils import load_iris_split
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt

# Multi-output model: sepal length & sepal width -> petal length & petal width
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_train[:, [0, 1]]  # sepal length and sepal width
y = X_train[:, [2, 3]]  # petal length and petal width

model = LinearRegression(batch_size=32, max_steps=100)
model.fit(X, y)
model.save("regression_multi_weights.npz")

plt.plot(model.loss_history)
plt.xlabel("Step Number")
plt.ylabel("Loss")
plt.title("Multi-Output: Sepal L,W -> Petal L,W - Training Loss")
plt.savefig("regression_multi_loss.png")

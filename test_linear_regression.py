import numpy as np
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression

iris = load_iris()

y = iris.data[:, 2]  # Petal length
X = np.delete(iris.data, 2, axis=1)  # Sepal length, sepal width, petal width

model = LinearRegression(batch_size= 32, max_steps= 100)
model.fit(X, y)
print(model.score(X, y))
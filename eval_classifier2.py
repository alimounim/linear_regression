from data_utils import load_iris_split
from LogisticRegression import LogisticRegression

# Classifier variant 2: sepal length/width -> species
X_train, X_test, y_train, y_test = load_iris_split()

X = X_test[:, [0, 1]]  # sepal length, sepal width
y = y_test

model = LogisticRegression()
model.load("classifier2_weights.npz")

accuracy = model.score(X, y)
print(f"Classifier 2 (sepal length/width) test accuracy: {accuracy:.4f}")

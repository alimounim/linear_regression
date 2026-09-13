from data_utils import load_iris_split
from LogisticRegression import LogisticRegression

# Classifier variant 1: petal length/width -> species
X_train, X_test, y_train, y_test = load_iris_split()

X = X_test[:, [2, 3]]  # petal length, petal width
y = y_test

model = LogisticRegression()
model.load("classifier1_weights.npz")

accuracy = model.score(X, y)
print(f"Classifier 1 (petal length/width) test accuracy: {accuracy:.4f}")

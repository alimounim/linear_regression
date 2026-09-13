from data_utils import load_iris_split
from LogisticRegression import LogisticRegression

# Classifier variant 3: all 4 features -> species
X_train, X_test, y_train, y_test = load_iris_split()

X = X_test  # sepal length, sepal width, petal length, petal width
y = y_test

model = LogisticRegression()
model.load("classifier3_weights.npz")

accuracy = model.score(X, y)
print(f"Classifier 3 (all features) test accuracy: {accuracy:.4f}")

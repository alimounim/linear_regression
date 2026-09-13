from data_utils import load_iris_split
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt

# Classifier variant 3: all 4 features -> species
X_train, X_test, y_train, y_test = load_iris_split()

X = X_train  # sepal length, sepal width, petal length, petal width
y = y_train

model = LogisticRegression(batch_size=32, patience=15, max_epochs=300)
model.fit(X, y)
model.save("classifier3_weights.npz")

print(f"Classifier 3 (all features) train accuracy: {model.score(X, y):.4f}")

# No decision-region plot here - that visualization only applies to the
# 2-feature variants (classifier 1 and 2), since it requires a 2D input space.
plt.figure()
plt.plot(model.loss_history)
plt.xlabel("Step Number")
plt.ylabel("Cross-Entropy Loss")
plt.title("Classifier 3 (All Features) - Training Loss")
plt.savefig("classifier3_loss.png")

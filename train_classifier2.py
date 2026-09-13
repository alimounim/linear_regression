from data_utils import load_iris_split
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Classifier variant 2: sepal length/width -> species
X_train, X_test, y_train, y_test = load_iris_split()

X = X_train[:, [0, 1]]  # sepal length, sepal width
y = y_train

model = LogisticRegression(batch_size=32, patience=15, max_epochs=300)
model.fit(X, y)
model.save("classifier2_weights.npz")

print(f"Classifier 2 (sepal length/width) train accuracy: {model.score(X, y):.4f}")

plt.figure()
plot_decision_regions(X, y, clf=model)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Logistic Regression Decision Regions: Sepal Length/Width")
plt.savefig("classifier2_decision_regions.png")

plt.figure()
plt.plot(model.loss_history)
plt.xlabel("Step Number")
plt.ylabel("Cross-Entropy Loss")
plt.title("Classifier 2 (Sepal L/W) - Training Loss")
plt.savefig("classifier2_loss.png")

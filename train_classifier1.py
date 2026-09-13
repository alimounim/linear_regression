from data_utils import load_iris_split
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Classifier variant 1: petal length/width -> species
X_train, X_test, y_train, y_test = load_iris_split()

X = X_train[:, [2, 3]]  # petal length, petal width
y = y_train

model = LogisticRegression(batch_size=32, patience=15, max_epochs=300)
model.fit(X, y)
model.save("classifier1_weights.npz")

print(f"Classifier 1 (petal length/width) train accuracy: {model.score(X, y):.4f}")

plt.figure()
plot_decision_regions(X, y, clf=model)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Logistic Regression Decision Regions: Petal Length/Width")
plt.savefig("classifier1_decision_regions.png")

plt.figure()
plt.plot(model.loss_history)
plt.xlabel("Step Number")
plt.ylabel("Cross-Entropy Loss")
plt.title("Classifier 1 (Petal L/W) - Training Loss")
plt.savefig("classifier1_loss.png")

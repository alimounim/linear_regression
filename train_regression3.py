from data_utils import load_iris_split
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt


# Model 3: sepal length & sepal width -> petal length
X_train, X_test, y_train, y_test = load_iris_split(test_size=0.1, random_state=42)

X = X_train[:, [0, 1]] # sepal length and sepal width
y = X_train[:, [2]] # petal length


model = LinearRegression(batch_size= 32, max_steps= 100)
model.fit(X, y)
model.save("regression3_weights.npz")

# plot the loss history (y-axis) against step number (x-axis) using matplotlib, labels the axes/title, and saves the plot as a PNG file
plt.plot(model.loss_history)
plt.xlabel("Step Number")
plt.ylabel("Loss")
plt.title("Model 3: Sepal Length & Sepal Width -> Petal Length - Training Loss")
plt.savefig("regression3_loss.png")